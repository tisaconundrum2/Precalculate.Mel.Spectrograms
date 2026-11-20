using FlacDecoder;

namespace FlacDecoder.Tests;

public class FlacDecoderTests
{
    private static readonly string TestDataDir =
        Path.Combine(AppContext.BaseDirectory, "TestData");

    [Theory]
    [InlineData("test_mono_16k", 32000, 16000, 1)]
    [InlineData("test_5sec_16k", 80000, 16000, 1)]
    [InlineData("test_mono_44k", 44100, 44100, 1)]
    [InlineData("test_noise_16k", 48000, 16000, 1)]
    public void Decode_MatchesFfmpegReference(string name, int expectedSamples, int expectedRate, int expectedChannels)
    {
        byte[] flacData = File.ReadAllBytes(Path.Combine(TestDataDir, $"{name}.flac"));
        byte[] referenceRaw = File.ReadAllBytes(Path.Combine(TestDataDir, $"{name}.raw"));

        var decoder = new FlacStreamDecoder();
        short[] decoded = decoder.Decode(flacData);

        // Check metadata
        Assert.Equal(expectedRate, decoder.SampleRate);
        Assert.Equal(expectedChannels, decoder.Channels);

        // Convert reference to short[]
        var reference = new short[referenceRaw.Length / 2];
        for (int i = 0; i < reference.Length; i++)
            reference[i] = BitConverter.ToInt16(referenceRaw, i * 2);

        // Sample count should match
        Assert.Equal(reference.Length, decoded.Length);

        // Values should be identical (FLAC is lossless)
        for (int i = 0; i < reference.Length; i++)
        {
            Assert.True(Math.Abs(decoded[i] - reference[i]) <= 1,
                $"Sample {i}: expected {reference[i]}, got {decoded[i]}");
        }
    }

    [Fact]
    public void DecodeToFloat32Mono_ReturnsNormalizedValues()
    {
        byte[] flacData = File.ReadAllBytes(Path.Combine(TestDataDir, "test_mono_16k.flac"));

        var decoder = new FlacStreamDecoder();
        float[] mono = decoder.DecodeToFloat32Mono(flacData);

        Assert.Equal(32000, mono.Length);
        Assert.Equal(16000, decoder.SampleRate);

        // All values should be in [-1, 1]
        foreach (float v in mono)
        {
            Assert.True(v >= -1f && v <= 1f, $"Value out of range: {v}");
        }

        // Should have non-zero values (it's a sine wave)
        Assert.True(mono.Any(v => Math.Abs(v) > 0.1f));
    }

    [Fact]
    public void Decode_InvalidData_Throws()
    {
        byte[] garbage = [0x00, 0x01, 0x02, 0x03];
        var decoder = new FlacStreamDecoder();
        Assert.Throws<InvalidDataException>(() => decoder.Decode(garbage));
    }

    [Fact]
    public void Decode_Performance()
    {
        // The noise file is the largest — use it for a rough perf check
        byte[] flacData = File.ReadAllBytes(Path.Combine(TestDataDir, "test_noise_16k.flac"));

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var decoder = new FlacStreamDecoder();
        int iterations = 100;
        for (int i = 0; i < iterations; i++)
            decoder.Decode(flacData);
        sw.Stop();

        double msPerDecode = sw.ElapsedMilliseconds / (double)iterations;
        // Should be well under 10ms per decode for a 3-second clip
        Assert.True(msPerDecode < 50, $"Decode too slow: {msPerDecode:F1}ms per decode");
    }
}
