using FlacDecoder;
using Xunit.Abstractions;

namespace FlacDecoder.Tests;

public class FlacDebugTest(ITestOutputHelper output)
{
    [Fact]
    public void DebugFrameParsing()
    {
        byte[] data = File.ReadAllBytes(
            Path.Combine(AppContext.BaseDirectory, "TestData", "test_mono_16k.flac"));
        output.WriteLine($"File: {data.Length} bytes");

        // Skip metadata
        int pos = 4;
        while (pos < data.Length)
        {
            bool last = (data[pos] & 0x80) != 0;
            int len = (data[pos + 1] << 16) | (data[pos + 2] << 8) | data[pos + 3];
            pos += 4 + len;
            if (last) break;
        }
        output.WriteLine($"Audio starts at byte {pos}");

        // Try decoding and see where it fails
        var decoder = new FlacStreamDecoder();
        try
        {
            short[] result = decoder.Decode(data);
            output.WriteLine($"Decoded: {result.Length} samples, SR={decoder.SampleRate}, CH={decoder.Channels}, BPS={decoder.BitsPerSample}");
        }
        catch (Exception ex)
        {
            output.WriteLine($"Decode error: {ex}");
        }

        // Count frame syncs after metadata
        int syncs = 0;
        for (int i = pos; i < data.Length - 1; i++)
        {
            if (data[i] == 0xFF && (data[i + 1] & 0xFC) == 0xF8)
            {
                syncs++;
                output.WriteLine($"  Sync at {i}");
            }
        }
        output.WriteLine($"Total frame syncs: {syncs}");
        output.WriteLine($"Expected samples: 32000, block size ~2304, expected frames: ~14");
    }
}
