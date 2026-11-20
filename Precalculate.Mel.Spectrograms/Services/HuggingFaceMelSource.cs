using System.Runtime.CompilerServices;
using Microsoft.Extensions.Options;
using ParquetSharp;
using Precalculate.Mel.Spectrograms.Contracts;
using Precalculate.Mel.Spectrograms.Models;

namespace Precalculate.Mel.Spectrograms.Services;

/// <summary>
/// Reads HF dataset parquet files from mounted paths, decodes audio,
/// computes mel spectrograms, and yields MelSamples.
///
/// HF datasets mount at /data/{org}/{repo}. Parquet files live under
/// /data/{org}/{repo}/{config}/{split}/0000.parquet, etc.
/// Audio is stored as a struct column { "bytes": byte[], "path": string }.
/// </summary>
public sealed class HuggingFaceMelSource : IMelSource
{
    private const int TargetSampleRate = 16_000;
    private const int MinAudioSamples = 1_600;   // ~0.1s at 16kHz
    private const int MaxAudioSamples = 240_000;  // ~15s at 16kHz
    private const int BatchSize = 64;

    private readonly IOptions<MountOptions> _mountOptions;
    private readonly IOptions<RecipeOptions> _recipeOptions;
    private readonly ILogger<HuggingFaceMelSource> _log;
    private readonly MelTransform _mel = new();
    private readonly CharTokenizer _tokenizer = new();

    public HuggingFaceMelSource(
        IOptions<MountOptions> mountOptions,
        IOptions<RecipeOptions> recipeOptions,
        ILogger<HuggingFaceMelSource> log)
    {
        _mountOptions = mountOptions;
        _recipeOptions = recipeOptions;
        _log = log;
    }

    private int MelBins => _mel.MelBins;

    public async IAsyncEnumerable<MelSample> ReadAsync(
        [EnumeratorCancellation] CancellationToken ct)
    {
        var mounts = _mountOptions.Value;
        var recipe = _recipeOptions.Value;

        _log.LogInformation("Processing recipe '{Name}' with {Count} sources",
            recipe.Name, recipe.Sources.Count);

        int globalIndex = 0;

        foreach (var source in recipe.Sources)
        {
            ct.ThrowIfCancellationRequested();

            string audioCol = source.AudioColumn ?? ResolveAudioColumn(source.Name);
            string textCol = source.TextColumn ?? ResolveTextColumn(source.Name);

            var parquetFiles = FindParquetFiles(mounts, source);
            if (parquetFiles.Count == 0)
            {
                _log.LogWarning("No parquet files found for {Source}/{Split}",
                    source.Name, source.Split);
                continue;
            }

            _log.LogInformation("Reading {Source}/{Split}: {Files} parquet files, " +
                "audio='{Audio}', text='{Text}'",
                source.Name, source.Split, parquetFiles.Count, audioCol, textCol);

            int sourceCount = 0;
            foreach (string path in parquetFiles)
            {
                ct.ThrowIfCancellationRequested();

                MelSample[] batch;
                const int maxRetries = 3;
                bool success = false;
                batch = [];

                for (int attempt = 1; attempt <= maxRetries; attempt++)
                {
                    try
                    {
                        batch = await ReadParquetFile(path, audioCol, textCol,
                            source.MaxSamples.HasValue ? source.MaxSamples.Value - sourceCount : null,
                            globalIndex);
                        success = true;
                        break;
                    }
                    catch (Exception ex) when (attempt < maxRetries)
                    {
                        _log.LogWarning(ex,
                            "Failed to read {Path} (attempt {Attempt}/{Max}), retrying in {Delay}s",
                            path, attempt, maxRetries, attempt * 5);
                        await Task.Delay(attempt * 5000, ct);
                    }
                    catch (Exception ex)
                    {
                        _log.LogWarning(ex,
                            "Failed to read {Path} after {Max} attempts, skipping",
                            path, maxRetries);
                    }
                }

                if (!success) continue;

                foreach (var sample in batch)
                {
                    yield return sample;
                    sourceCount++;
                    globalIndex++;

                    if (sourceCount % 1000 == 0)
                        _log.LogInformation("    {Source}: {Count} samples processed ({Total} total)",
                            source.Name, sourceCount, globalIndex);
                }

                if (source.MaxSamples.HasValue && sourceCount >= source.MaxSamples.Value)
                    break;
            }

            _log.LogInformation("  {Source}/{Split}: {Count} samples",
                source.Name, source.Split, sourceCount);

            await Task.CompletedTask; // Keep async signature; yields are the real work
        }

        _log.LogInformation("Recipe '{Name}' complete: {Total} total samples",
            recipe.Name, globalIndex);
    }

    private async Task<MelSample[]> ReadParquetFile(
        string path, string audioCol, string textCol,
        int? remaining, int startIndex)
    {
        int globalIndex = startIndex;
        using var reader = new ParquetFileReader(path);
        var samples = new List<MelSample>();
        int decoded = 0, skipped = 0, totalRowsSoFar = 0;
        long totalRowsInFile = reader.FileMetaData.NumRows;
        var fileStart = System.Diagnostics.Stopwatch.StartNew();

        for (int rg = 0; rg < reader.FileMetaData.NumRowGroups; rg++)
        {
            using var rowGroup = reader.RowGroup(rg);
            int numRows = checked((int)rowGroup.MetaData.NumRows);

            // Log schema once per file for debugging
            if (rg == 0)
            {
                var schema = reader.FileMetaData.Schema;
                var cols = new string[schema.NumColumns];
                for (int c = 0; c < schema.NumColumns; c++)
                    cols[c] = schema.Column(c).Name;
                _log.LogDebug("Schema for {Path}: [{Columns}]", path, string.Join(", ", cols));
            }

            // Find column indices
            int audioColIdx = FindColumnIndex(reader.FileMetaData.Schema, audioCol);
            int textColIdx = FindColumnIndex(reader.FileMetaData.Schema, textCol);

            // Read text column
            var texts = ReadStringColumn(rowGroup, textColIdx, numRows);

            // Read audio bytes column (struct with "bytes" field)
            var audioBytes = ReadAudioBytesColumn(rowGroup, audioColIdx, numRows);

            // Collect valid rows, then decode audio in parallel batches
            var validRows = new List<(byte[] Audio, string Text, int[] TokenIds)>();

            for (int row = 0; row < numRows; row++)
            {
                if (remaining.HasValue && samples.Count + validRows.Count >= remaining.Value)
                    break;

                string? text = texts[row];
                byte[]? audio = audioBytes[row];

                if (string.IsNullOrWhiteSpace(text) || audio is null || audio.Length == 0)
                {
                    skipped++;
                    continue;
                }

                string normalized = CharTokenizer.Normalize(text);
                int[] tokenIds = _tokenizer.Encode(normalized);
                if (tokenIds.Length == 0)
                {
                    skipped++;
                    continue;
                }

                validRows.Add((audio, normalized, tokenIds));
            }

            // Decode audio in parallel (N concurrent ffmpeg processes)
            int parallelism = Math.Max(2, Environment.ProcessorCount);
            _log.LogInformation("    {File}: decoding {Count} clips with {P} parallel workers",
                Path.GetFileName(path), validRows.Count, parallelism);

            for (int batch = 0; batch < validRows.Count; batch += BatchSize)
            {
                int batchEnd = Math.Min(batch + BatchSize, validRows.Count);
                var batchSlice = validRows.GetRange(batch, batchEnd - batch);

                // Decode FLAC + compute mel in parallel across all cores
                var processTasks = batchSlice.Select((r, idx) => Task.Run(() =>
                {
                    var pcm = DecodeAudio(r.Audio);
                    if (pcm is null || pcm.Length < MinAudioSamples || pcm.Length > MaxAudioSamples)
                        return ((float[]?, int, int))default;

                    // Each task gets its own MelTransform (they're cheap, avoid contention)
                    var mel = new MelTransform();
                    var (values, frames) = mel.Compute(pcm);
                    return (values, frames, idx);
                })).ToArray();

                (float[]? Values, int Frames, int Idx)[] results;
                try
                {
                    results = await Task.WhenAll(processTasks);
                }
                catch
                {
                    results = new (float[]?, int, int)[processTasks.Length];
                    for (int i = 0; i < processTasks.Length; i++)
                    {
                        try { results[i] = await processTasks[i]; }
                        catch { results[i] = default; }
                    }
                }

                // Collect results in order
                for (int i = 0; i < batchSlice.Count; i++)
                {
                    var (values, frames, _) = results[i];
                    if (values is null || frames <= 0)
                    {
                        skipped++;
                        continue;
                    }

                    decoded++;
                    string sampleId = $"{globalIndex:D8}";
                    samples.Add(new MelSample(
                        Id: sampleId,
                        MelBins: MelBins,
                        Frames: frames,
                        Values: values,
                        TokenIds: batchSlice[i].TokenIds));
                    globalIndex++;
                }

                // Progress
                int currentRow = totalRowsSoFar + Math.Min(batchEnd, numRows);
                double pct = 100.0 * currentRow / totalRowsInFile;
                double elapsed = fileStart.Elapsed.TotalSeconds;
                double rate = decoded / Math.Max(elapsed, 0.1);
                _log.LogInformation("    {File}: row {Row}/{Total} ({Pct:F0}%), " +
                    "{Decoded} decoded, {Skipped} skipped, {Rate:F0} samples/s",
                    Path.GetFileName(path), currentRow, totalRowsInFile, pct,
                    decoded, skipped, rate);
            }

            totalRowsSoFar += numRows;
        }

        fileStart.Stop();
        _log.LogInformation("  Finished {File}: {Samples} samples in {Secs:F1}s " +
            "({Decoded} decoded, {Skipped} skipped)",
            Path.GetFileName(path), samples.Count, fileStart.Elapsed.TotalSeconds,
            decoded, skipped);

        return samples.ToArray();
    }

    /// <summary>
    /// Decode audio bytes (FLAC or WAV) to 16kHz mono float32 PCM.
    /// Auto-detects format from magic bytes.
    /// </summary>
    [ThreadStatic] private static FlacDecoder.FlacStreamDecoder? _flacDecoder;

    private static float[]? DecodeAudio(byte[] audioBytes)
    {
        if (audioBytes.Length < 4) return null;

        try
        {
            // Detect format from magic bytes
            if (audioBytes[0] == (byte)'f' && audioBytes[1] == (byte)'L' &&
                audioBytes[2] == (byte)'a' && audioBytes[3] == (byte)'C')
            {
                // FLAC
                _flacDecoder ??= new FlacDecoder.FlacStreamDecoder();
                float[] pcm = _flacDecoder.DecodeToFloat32Mono(audioBytes);
                if (_flacDecoder.SampleRate != TargetSampleRate)
                    pcm = Resample(pcm, _flacDecoder.SampleRate, TargetSampleRate);
                return pcm;
            }

            if (audioBytes[0] == (byte)'R' && audioBytes[1] == (byte)'I' &&
                audioBytes[2] == (byte)'F' && audioBytes[3] == (byte)'F')
            {
                // WAV — parse header and extract PCM
                return DecodeWav(audioBytes);
            }

            return null; // Unknown format
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Decode a WAV file to mono float32 PCM. Handles pcm_f32le, pcm_s16le, pcm_s24le, pcm_s32le.
    /// </summary>
    private static float[]? DecodeWav(byte[] wav)
    {
        if (wav.Length < 44) return null;

        // Find "fmt " chunk
        int pos = 12; // Skip RIFF header
        int fmtPos = -1, dataPos = -1, dataSize = 0;
        ushort audioFormat = 0, numChannels = 0;
        int sampleRate = 0;
        ushort bitsPerSample = 0;

        while (pos + 8 <= wav.Length)
        {
            string chunkId = System.Text.Encoding.ASCII.GetString(wav, pos, 4);
            int chunkSize = BitConverter.ToInt32(wav, pos + 4);

            if (chunkId == "fmt ")
            {
                fmtPos = pos + 8;
                audioFormat = BitConverter.ToUInt16(wav, fmtPos);
                numChannels = BitConverter.ToUInt16(wav, fmtPos + 2);
                sampleRate = BitConverter.ToInt32(wav, fmtPos + 4);
                bitsPerSample = BitConverter.ToUInt16(wav, fmtPos + 14);
            }
            else if (chunkId == "data")
            {
                dataPos = pos + 8;
                dataSize = chunkSize;
                break;
            }

            pos += 8 + chunkSize;
            if (chunkSize % 2 != 0) pos++; // Padding byte
        }

        if (fmtPos < 0 || dataPos < 0 || dataSize == 0) return null;

        int bytesPerSample = bitsPerSample / 8;
        int totalSamples = dataSize / bytesPerSample / numChannels;
        var result = new float[totalSamples];

        for (int i = 0; i < totalSamples; i++)
        {
            float sum = 0;
            for (int ch = 0; ch < numChannels; ch++)
            {
                int offset = dataPos + (i * numChannels + ch) * bytesPerSample;
                if (offset + bytesPerSample > wav.Length) break;

                float val = (audioFormat, bitsPerSample) switch
                {
                    (3, 32) => BitConverter.ToSingle(wav, offset),          // pcm_f32le (VoxPopuli, AMI)
                    (1, 16) => BitConverter.ToInt16(wav, offset) / 32768f,  // pcm_s16le
                    (1, 32) => BitConverter.ToInt32(wav, offset) / 2147483648f,
                    (1, 24) => ((wav[offset] | (wav[offset + 1] << 8) | (wav[offset + 2] << 16))
                               - (wav[offset + 2] >= 128 ? 16777216 : 0)) / 8388608f,
                    (1, 8) => (wav[offset] - 128) / 128f,
                    _ => 0f
                };
                sum += val;
            }
            result[i] = sum / numChannels;
        }

        // Resample if needed
        if (sampleRate != TargetSampleRate)
            result = Resample(result, sampleRate, TargetSampleRate);

        return result;
    }

    private static float[] Resample(float[] input, int fromRate, int toRate)
    {
        double ratio = (double)fromRate / toRate;
        int outLen = (int)(input.Length / ratio);
        var output = new float[outLen];
        for (int i = 0; i < outLen; i++)
        {
            double srcIdx = i * ratio;
            int idx = (int)srcIdx;
            float frac = (float)(srcIdx - idx);
            output[i] = idx + 1 < input.Length
                ? input[idx] * (1f - frac) + input[idx + 1] * frac
                : input[idx];
        }
        return output;
    }

    private List<string> FindParquetFiles(MountOptions mounts, RecipeSource source)
    {
        // HF datasets mount the full repo at basePath/org/repo.
        // Parquet layout varies by dataset:
        //   Pattern A: {config}/{split}/0000.parquet        (LibriSpeech "all")
        //   Pattern B: {config}/{split}-00000-of-00030.parquet (VoxPopuli, People's Speech)
        //   Pattern C: {split}/0000.parquet                 (no config)
        //   Pattern D: data/{split}-*.parquet               (some datasets)

        string mountPath = mounts.ResolvePath(new DatasetMount
        {
            Name = source.Name,
            RepoId = source.Name,
        });

        // Log what we can see at the mount for debugging
        if (Directory.Exists(mountPath))
        {
            var topLevel = Directory.GetDirectories(mountPath)
                .Select(Path.GetFileName)
                .Take(10);
            _log.LogInformation("  Mount {Path} contains dirs: [{Dirs}]",
                mountPath, string.Join(", ", topLevel));
        }
        else
        {
            _log.LogWarning("  Mount path does not exist: {Path}", mountPath);
            return [];
        }

        var searchPaths = new List<(string Dir, bool IsSplitDir)>();

        // Pattern A: mount/config/split/ (files named 0000.parquet)
        if (source.Config is not null)
        {
            searchPaths.Add((Path.Combine(mountPath, source.Config, source.Split), true));
            // Pattern B: mount/config/ (files named {split}-00000-of-*.parquet)
            searchPaths.Add((Path.Combine(mountPath, source.Config), false));
        }

        // Pattern C: mount/split/
        searchPaths.Add((Path.Combine(mountPath, source.Split), true));

        // Pattern D: mount/data/ with split prefix
        searchPaths.Add((Path.Combine(mountPath, "data"), false));

        // Fallback: mount/ directly
        searchPaths.Add((mountPath, false));

        // Normalize split name for filename matching: "train.clean.100" → "train.clean.100" and "train-clean-100"
        string splitDot = source.Split;
        string splitDash = source.Split.Replace(".", "-");

        var files = new List<string>();
        foreach (var (dir, isSplitDir) in searchPaths)
        {
            if (!Directory.Exists(dir))
                continue;

            var candidates = Directory.GetFiles(dir, "*.parquet")
                .Where(f =>
                {
                    // In a split-specific directory, accept all parquet files
                    if (isSplitDir)
                        return true;

                    string name = Path.GetFileName(f);
                    // Match: {split}-00000-of-00030.parquet or {split}.parquet or 0000.parquet
                    return name.StartsWith(splitDash, StringComparison.OrdinalIgnoreCase)
                        || name.StartsWith(splitDot, StringComparison.OrdinalIgnoreCase);
                })
                .OrderBy(f => f)
                .ToList();

            if (candidates.Count > 0)
            {
                _log.LogInformation("  Found {Count} parquet files in {Dir}",
                    candidates.Count, dir);
                files.AddRange(candidates);
                break;
            }
        }

        return files;
    }

    /// <summary>
    /// Find a column index in the flattened parquet schema.
    /// HF datasets have varying layouts:
    ///   - Nested struct: audio.bytes, audio.path (VoxPopuli, People's Speech)
    ///   - Flattened: bytes, path (LibriSpeech)
    /// For "audio" we try: audio → audio.bytes → bytes (fallback for flattened schemas).
    /// </summary>
    private int FindColumnIndex(SchemaDescriptor schema, string name)
    {
        var colNames = new string[schema.NumColumns];
        for (int i = 0; i < schema.NumColumns; i++)
            colNames[i] = schema.Column(i).Name;

        // Exact match
        for (int i = 0; i < schema.NumColumns; i++)
        {
            if (colNames[i].Equals(name, StringComparison.OrdinalIgnoreCase))
                return i;
        }

        // HF Audio struct nested: "{name}.bytes"
        string bytesNested = $"{name}.bytes";
        for (int i = 0; i < schema.NumColumns; i++)
        {
            if (colNames[i].Equals(bytesNested, StringComparison.OrdinalIgnoreCase))
                return i;
        }

        // HF Audio flattened: the struct is exploded to top-level "bytes" column
        // This happens when looking for "audio" but the schema has "bytes" directly
        if (name.Equals("audio", StringComparison.OrdinalIgnoreCase))
        {
            for (int i = 0; i < schema.NumColumns; i++)
            {
                if (colNames[i].Equals("bytes", StringComparison.OrdinalIgnoreCase))
                    return i;
            }
        }

        // Partial match: column path contains the name
        for (int i = 0; i < schema.NumColumns; i++)
        {
            if (colNames[i].Contains(name, StringComparison.OrdinalIgnoreCase))
                return i;
        }

        _log.LogError("Column '{Name}' not found. Schema columns: [{Columns}]",
            name, string.Join(", ", colNames));
        throw new InvalidDataException($"Column '{name}' not found in parquet schema.");
    }

    private static string?[] ReadStringColumn(RowGroupReader rowGroup, int colIdx, int numRows)
    {
        using var colReader = rowGroup.Column(colIdx).LogicalReader<string?>();
        var values = new string?[numRows];
        int read = colReader.ReadBatch(values);
        return values;
    }

    private static byte[]?[] ReadAudioBytesColumn(RowGroupReader rowGroup, int colIdx, int numRows)
    {
        // HF stores audio as a struct { bytes: byte[], path: string }
        // In parquet this is a group column. We need the "bytes" child.
        // ParquetSharp reads nested columns via the flattened schema.
        // The bytes sub-column is at colIdx (the struct) or colIdx+offset for "bytes".

        // Try reading directly as byte[] first (some datasets flatten it)
        try
        {
            using var colReader = rowGroup.Column(colIdx).LogicalReader<byte[]?>();
            var values = new byte[]?[numRows];
            colReader.ReadBatch(values);
            return values!;
        }
        catch
        {
            // If that fails, the audio column is likely a struct.
            // Try the next column which should be the "bytes" field.
            try
            {
                using var colReader = rowGroup.Column(colIdx + 1).LogicalReader<byte[]?>();
                var values = new byte[]?[numRows];
                colReader.ReadBatch(values);
                return values!;
            }
            catch
            {
                return new byte[]?[numRows];
            }
        }
    }

    private static string ResolveAudioColumn(string datasetName) => datasetName switch
    {
        _ when datasetName.Contains("librispeech") => "audio",
        _ when datasetName.Contains("voxpopuli") => "audio",
        _ when datasetName.Contains("peoples_speech") => "audio",
        _ when datasetName.Contains("common_voice") => "audio",
        _ => "audio"
    };

    private static string ResolveTextColumn(string datasetName) => datasetName switch
    {
        _ when datasetName.Contains("librispeech") => "text",
        _ when datasetName.Contains("voxpopuli") => "raw_text",
        _ when datasetName.Contains("peoples_speech") => "text",
        _ when datasetName.Contains("common_voice") => "sentence",
        _ => "text"
    };
}
