using System.Diagnostics;
using Precalculate.Mel.Spectrograms.Contracts;

namespace Precalculate.Mel.Spectrograms.Services;

/// <summary>
/// Publishes completed shards to a HuggingFace Dataset repo using huggingface-cli.
/// Shells out to the CLI which handles LFS, retries, and the commit API internally.
/// This is the same proven upload path that the Python cache_repository uses via HfApi.
/// </summary>
public sealed class HubShardPublisher : IShardPublisher
{
    private readonly string _repoId;
    private readonly string _pathPrefix;
    private readonly string _token;
    private readonly ILogger<HubShardPublisher> _log;

    public HubShardPublisher(
        string repoId,
        string? pathPrefix,
        string token,
        ILogger<HubShardPublisher> log)
    {
        _repoId = repoId;
        _pathPrefix = pathPrefix ?? "";
        _token = token;
        _log = log;
    }

    public async Task PublishAsync(
        string readyPath,
        string shardName,
        CancellationToken ct)
    {
        string pathInRepo = string.IsNullOrEmpty(_pathPrefix)
            ? shardName
            : $"{_pathPrefix}/{shardName}";

        long fileSize = new FileInfo(readyPath).Length;
        _log.LogInformation("Uploading {Shard} ({Size:F1} MB) to {Repo}/{Path}",
            shardName, fileSize / 1_000_000.0, _repoId, pathInRepo);

        const int maxRetries = 3;
        for (int attempt = 1; attempt <= maxRetries; attempt++)
        {
            try
            {
                var result = await RunHfUpload(readyPath, pathInRepo, ct);
                if (result == 0)
                {
                    _log.LogInformation("Uploaded {Shard} successfully ({Size:F0} MB)",
                        shardName, fileSize / 1_000_000.0);
                    return;
                }

                _log.LogWarning("huggingface-cli upload failed with exit code {Code} (attempt {Attempt}/{Max})",
                    result, attempt, maxRetries);
            }
            catch (Exception ex) when (attempt < maxRetries)
            {
                _log.LogWarning(ex, "Upload {Shard} failed (attempt {Attempt}/{Max})",
                    shardName, attempt, maxRetries);
            }

            if (attempt < maxRetries)
            {
                int delay = attempt * 15;
                _log.LogInformation("Retrying in {Delay}s", delay);
                await Task.Delay(delay * 1000, ct);
            }
        }

        _log.LogError("Failed to upload {Shard} after {Max} attempts. File preserved at {Path}",
            shardName, maxRetries, readyPath);
        throw new IOException($"Failed to upload {shardName} after {maxRetries} attempts");
    }

    private async Task<int> RunHfUpload(string localPath, string pathInRepo, CancellationToken ct)
    {
        // huggingface-cli upload <repo_id> <local_path> <path_in_repo> --repo-type dataset
        var psi = new ProcessStartInfo
        {
            FileName = "python3",
            ArgumentList =
            {
                "-c",
                $"from huggingface_hub import HfApi; " +
                $"HfApi().upload_file(" +
                $"path_or_fileobj='{localPath}', " +
                $"path_in_repo='{pathInRepo}', " +
                $"repo_id='{_repoId}', " +
                $"repo_type='dataset', " +
                $"commit_message='Add {Path.GetFileName(localPath)}')"
            },
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };

        // Pass token via environment
        psi.Environment["HF_TOKEN"] = _token;

        using var process = Process.Start(psi)!;

        // Stream output for visibility
        var stdoutTask = Task.Run(async () =>
        {
            string? line;
            while ((line = await process.StandardOutput.ReadLineAsync(ct)) is not null)
                _log.LogDebug("  hf-cli: {Line}", line);
        }, ct);

        var stderrTask = Task.Run(async () =>
        {
            string? line;
            while ((line = await process.StandardError.ReadLineAsync(ct)) is not null)
                _log.LogDebug("  hf-cli err: {Line}", line);
        }, ct);

        await process.WaitForExitAsync(ct);
        await Task.WhenAll(stdoutTask, stderrTask);

        return process.ExitCode;
    }
}
