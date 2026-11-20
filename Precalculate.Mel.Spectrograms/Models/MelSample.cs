namespace Precalculate.Mel.Spectrograms.Models;

public sealed record MelSample(
    string Id,
    int MelBins,
    int Frames,
    ReadOnlyMemory<float> Values,
    ReadOnlyMemory<int> TokenIds);
