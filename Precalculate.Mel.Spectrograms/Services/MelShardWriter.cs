using System.Formats.Tar;
using System.Runtime.InteropServices;
using System.Text.Json;

using Precalculate.Mel.Spectrograms.Models;

namespace Precalculate.Mel.Spectrograms.Services;

public sealed class MelShardWriter : IDisposable
{
    private readonly FileStream _file;
    private readonly TarWriter _tar;

    public int Count { get; private set; }
    public long MelBytes { get; private set; }

    public MelShardWriter(string path)
    {
        _file = new FileStream(
            path,
            FileMode.CreateNew,
            FileAccess.Write,
            FileShare.None,
            bufferSize: 256 * 1024,
            options: FileOptions.SequentialScan);

        _tar = new TarWriter(_file, TarEntryFormat.Ustar, leaveOpen: true);
    }

    public void Append(MelSample sample)
    {
        if (!BitConverter.IsLittleEndian)
            throw new PlatformNotSupportedException("Expected little-endian.");

        if (sample.MelBins <= 0 || sample.Frames <= 0 ||
            (long)sample.MelBins * sample.Frames != sample.Values.Length)
        {
            throw new InvalidDataException($"Invalid mel shape: {sample.Id}");
        }

        string key = Count.ToString("D6");

        byte[] mel = MemoryMarshal.AsBytes(sample.Values.Span).ToArray();
        byte[] metadata = JsonSerializer.SerializeToUtf8Bytes(new
        {
            version = 1,
            id = sample.Id,
            shape = new[] { sample.MelBins, sample.Frames },
            dtype = "<f4",
            token_ids = sample.TokenIds.ToArray()
        });

        WriteEntry($"{key}.f32", mel);
        WriteEntry($"{key}.json", metadata);

        MelBytes += mel.Length;
        Count++;
    }

    private void WriteEntry(string name, byte[] contents)
    {
        using var data = new MemoryStream(contents, writable: false);
        var entry = new UstarTarEntry(TarEntryType.RegularFile, name)
        {
            DataStream = data,
            ModificationTime = DateTimeOffset.UnixEpoch
        };
        _tar.WriteEntry(entry);
    }

    public void Dispose()
    {
        try
        {
            _tar.Dispose();
            _file.Flush(flushToDisk: true);
        }
        finally
        {
            _file.Dispose();
        }
    }
}
