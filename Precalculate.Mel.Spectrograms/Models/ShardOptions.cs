namespace Precalculate.Mel.Spectrograms.Models;

public sealed class ShardOptions
{
    public const string SectionName = "Shard";

    public required string SpoolDirectory { get; init; }
    public int SamplesPerShard { get; init; } = 5_000;
    public long MaxMelBytes { get; init; } = 2_000_000_000;
}
