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
public sealed class HuggingFaceMelSource(
    IOptions<MountOptions> mountOptions,
    IOptions<RecipeOptions> recipeOptions,
    ILogger<HuggingFaceMelSource> log) : IMelSource
{
    private const int TargetSampleRate = 16_000;
    private const int MinAudioSamples = 1_600;   // ~0.1s at 16kHz
    private const int MaxAudioSamples = 240_000;  // ~15s at 16kHz

    private const int BatchSize = 64; // Audio clips to decode before GPU mel batch

    private readonly GpuMelTransform _mel = new();
    private readonly CharTokenizer _tokenizer = new();

    public async IAsyncEnumerable<MelSample> ReadAsync(
        [EnumeratorCancellation] CancellationToken ct)
    {
        var mounts = mountOptions.Value;
        var recipe = recipeOptions.Value;

        log.LogInformation("Using {Device} for mel spectrograms",
            _mel.IsGpu ? "CUDA GPU" : "CPU");
        log.LogInformation("Processing recipe '{Name}' with {Count} sources",
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
                log.LogWarning("No parquet files found for {Source}/{Split}",
                    source.Name, source.Split);
                continue;
            }

            log.LogInformation("Reading {Source}/{Split}: {Files} parquet files, " +
                "audio='{Audio}', text='{Text}'",
                source.Name, source.Split, parquetFiles.Count, audioCol, textCol);

            int sourceCount = 0;
            foreach (string path in parquetFiles)
            {
                ct.ThrowIfCancellationRequested();

                MelSample[] batch;
                try
                {
                    batch = ReadParquetFile(path, audioCol, textCol,
                        source.MaxSamples.HasValue ? source.MaxSamples.Value - sourceCount : null,
                        globalIndex);
                }
                catch (Exception ex)
                {
                    log.LogWarning(ex, "Failed to read {Path}, skipping", path);
                    continue;
                }

                foreach (var sample in batch)
                {
                    yield return sample;
                    sourceCount++;
                    globalIndex++;

                    if (sourceCount % 1000 == 0)
                        log.LogInformation("    {Source}: {Count} samples processed ({Total} total)",
                            source.Name, sourceCount, globalIndex);
                }

                if (source.MaxSamples.HasValue && sourceCount >= source.MaxSamples.Value)
                    break;
            }

            log.LogInformation("  {Source}/{Split}: {Count} samples",
                source.Name, source.Split, sourceCount);

            await Task.CompletedTask; // Keep async signature; yields are the real work
        }

        log.LogInformation("Recipe '{Name}' complete: {Total} total samples",
            recipe.Name, globalIndex);
    }

    private MelSample[] ReadParquetFile(
        string path, string audioCol, string textCol,
        int? remaining, int startIndex)
    {
        int globalIndex = startIndex;
        using var reader = new ParquetFileReader(path);
        var samples = new List<MelSample>();

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
                log.LogDebug("Schema for {Path}: [{Columns}]", path, string.Join(", ", cols));
            }

            // Find column indices
            int audioColIdx = FindColumnIndex(reader.FileMetaData.Schema, audioCol);
            int textColIdx = FindColumnIndex(reader.FileMetaData.Schema, textCol);

            // Read text column
            var texts = ReadStringColumn(rowGroup, textColIdx, numRows);

            // Read audio bytes column (struct with "bytes" field)
            var audioBytes = ReadAudioBytesColumn(rowGroup, audioColIdx, numRows);

            // Collect decoded audio + metadata in batches for GPU mel processing
            var pending = new List<(float[] Pcm, string Text, int[] TokenIds)>();

            void FlushBatch()
            {
                if (pending.Count == 0) return;

                var pcmBatch = pending.Select(p => p.Pcm).ToArray();
                var melResults = _mel.ComputeBatch(pcmBatch);

                for (int b = 0; b < pending.Count; b++)
                {
                    var (values, frames) = melResults[b];
                    if (values is null || frames <= 0) continue;

                    string sampleId = $"{globalIndex:D8}";
                    samples.Add(new MelSample(
                        Id: sampleId,
                        MelBins: _mel.MelBins,
                        Frames: frames,
                        Values: values,
                        TokenIds: pending[b].TokenIds));
                    globalIndex++;
                }
                pending.Clear();
            }

            for (int row = 0; row < numRows; row++)
            {
                if (remaining.HasValue && samples.Count >= remaining.Value)
                    break;

                string? text = texts[row];
                byte[]? audio = audioBytes[row];

                if (string.IsNullOrWhiteSpace(text) || audio is null || audio.Length == 0)
                    continue;

                string normalized = CharTokenizer.Normalize(text);
                int[] tokenIds = _tokenizer.Encode(normalized);
                if (tokenIds.Length == 0)
                    continue;

                float[]? pcm;
                try
                {
                    pcm = DecodeAudio(audio);
                }
                catch
                {
                    continue;
                }

                if (pcm is null || pcm.Length < MinAudioSamples || pcm.Length > MaxAudioSamples)
                    continue;

                pending.Add((pcm, normalized, tokenIds));

                if (pending.Count >= BatchSize)
                    FlushBatch();
            }

            // Flush remaining partial batch
            FlushBatch();
        }

        return samples.ToArray();
    }

    /// <summary>
    /// Decode any audio format (FLAC, WAV, MP3, OGG, etc.) to 16kHz mono float32 PCM
    /// by piping bytes through ffmpeg. HF parquet stores audio as FLAC.
    /// </summary>
    private static float[]? DecodeAudio(byte[] audioBytes)
    {
        using var process = new System.Diagnostics.Process();
        process.StartInfo = new System.Diagnostics.ProcessStartInfo
        {
            FileName = "ffmpeg",
            // Read from stdin, output raw 16-bit LE mono 16kHz PCM to stdout
            Arguments = "-hide_banner -loglevel error -i pipe:0 -f s16le -acodec pcm_s16le -ar 16000 -ac 1 pipe:1",
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };

        process.Start();

        // Write audio bytes to stdin, then close it so ffmpeg knows input is done
        var writeTask = Task.Run(() =>
        {
            process.StandardInput.BaseStream.Write(audioBytes, 0, audioBytes.Length);
            process.StandardInput.BaseStream.Close();
        });

        // Read raw PCM from stdout
        using var outMs = new MemoryStream();
        process.StandardOutput.BaseStream.CopyTo(outMs);

        writeTask.Wait();
        process.WaitForExit();

        if (process.ExitCode != 0)
            return null;

        var pcm16 = outMs.ToArray();
        if (pcm16.Length < 2)
            return null;

        // Convert s16le to float32
        var pcmFloat = new float[pcm16.Length / 2];
        for (int i = 0; i < pcmFloat.Length; i++)
        {
            short sample = BitConverter.ToInt16(pcm16, i * 2);
            pcmFloat[i] = sample / 32768f;
        }
        return pcmFloat;
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
            log.LogInformation("  Mount {Path} contains dirs: [{Dirs}]",
                mountPath, string.Join(", ", topLevel));
        }
        else
        {
            log.LogWarning("  Mount path does not exist: {Path}", mountPath);
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
                log.LogInformation("  Found {Count} parquet files in {Dir}",
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

        log.LogError("Column '{Name}' not found. Schema columns: [{Columns}]",
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
