using System.Net.Http.Headers;
using Precalculate.Mel.Spectrograms.Contracts;

namespace Precalculate.Mel.Spectrograms.Services;

/// <summary>
/// Publishes completed shards to a HuggingFace Dataset repo via the Hub API.
/// Each shard is uploaded as a single file commit. On failure, the local file
/// is preserved so publication can be retried.
/// </summary>
public sealed class HubShardPublisher : IShardPublisher
{
    private readonly string _repoId;
    private readonly string _pathPrefix;
    private readonly string _token;
    private readonly HttpClient _http;
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
        _http = new HttpClient
        {
            Timeout = TimeSpan.FromMinutes(30) // Large shard uploads
        };
        _http.DefaultRequestHeaders.Authorization =
            new AuthenticationHeaderValue("Bearer", _token);
    }

    public async Task PublishAsync(
        string readyPath,
        string shardName,
        CancellationToken ct)
    {
        string pathInRepo = string.IsNullOrEmpty(_pathPrefix)
            ? shardName
            : $"{_pathPrefix}/{shardName}";

        _log.LogInformation("Uploading {Shard} ({Size:F1} MB) to {Repo}/{Path}",
            shardName,
            new FileInfo(readyPath).Length / 1_000_000.0,
            _repoId,
            pathInRepo);

        // Use the HF Hub upload file API
        // POST https://huggingface.co/api/datasets/{repo_id}/upload/{revision}/{path_in_repo}
        string url = $"https://huggingface.co/api/datasets/{_repoId}/upload/main/{Uri.EscapeDataString(pathInRepo)}";

        const int maxRetries = 3;
        for (int attempt = 1; attempt <= maxRetries; attempt++)
        {
            try
            {
                await using var fileStream = File.OpenRead(readyPath);
                using var content = new StreamContent(fileStream);
                content.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");

                using var response = await _http.PostAsync(url, content, ct);

                if (response.IsSuccessStatusCode)
                {
                    _log.LogInformation("Uploaded {Shard} successfully", shardName);
                    return;
                }

                string body = await response.Content.ReadAsStringAsync(ct);
                _log.LogWarning("Upload {Shard} failed (HTTP {Status}): {Body}",
                    shardName, (int)response.StatusCode, body);

                if (attempt < maxRetries)
                {
                    int delay = attempt * 10;
                    _log.LogInformation("Retrying in {Delay}s (attempt {Attempt}/{Max})",
                        delay, attempt, maxRetries);
                    await Task.Delay(delay * 1000, ct);
                }
            }
            catch (Exception ex) when (attempt < maxRetries)
            {
                int delay = attempt * 10;
                _log.LogWarning(ex, "Upload {Shard} failed (attempt {Attempt}/{Max}), retrying in {Delay}s",
                    shardName, attempt, maxRetries, delay);
                await Task.Delay(delay * 1000, ct);
            }
        }

        // Final attempt failure — leave the file for manual retry
        _log.LogError("Failed to upload {Shard} after {Max} attempts. File preserved at {Path}",
            shardName, maxRetries, readyPath);
        throw new IOException($"Failed to upload {shardName} after {maxRetries} attempts");
    }
}
