namespace Precalculate.Mel.Spectrograms.Models;

public sealed class MountOptions
{
    public const string SectionName = "Mounts";

    /// <summary>
    /// Base path where HF datasets are mounted. Defaults to /data for HF Jobs.
    /// </summary>
    public string BasePath { get; init; } = "/data";

    /// <summary>
    /// Output cache repository mount (read-write).
    /// </summary>
    public DatasetMount Output { get; init; } = default!;

    /// <summary>
    /// Input dataset mounts (read-only).
    /// </summary>
    public List<DatasetMount> Inputs { get; init; } = [];

    /// <summary>
    /// Resolves the full filesystem path for a mount.
    /// </summary>
    public string ResolvePath(DatasetMount mount)
        => string.IsNullOrEmpty(mount.MountPath)
            ? Path.Combine(BasePath, mount.RepoId.Replace('/', Path.DirectorySeparatorChar))
            : mount.MountPath;
}
