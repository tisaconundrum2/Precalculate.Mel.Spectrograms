namespace Precalculate.Mel.Spectrograms.Models;

public sealed class DatasetMount
{
    public required string Name { get; init; }
    public required string RepoId { get; init; }
    public string MountPath { get; init; } = default!;
    public bool ReadOnly { get; init; } = true;
}
