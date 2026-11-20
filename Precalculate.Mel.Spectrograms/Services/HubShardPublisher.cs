using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Precalculate.Mel.Spectrograms.Contracts;

namespace Precalculate.Mel.Spectrograms.Services;

/// <summary>
/// Publishes completed shards to a HuggingFace Dataset repo via the commit API.
/// Uses LFS for large files (&gt;10MB). On failure, the local file is preserved.
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
            Timeout = TimeSpan.FromMinutes(60)
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

        long fileSize = new FileInfo(readyPath).Length;
        _log.LogInformation("Uploading {Shard} ({Size:F1} MB) to {Repo}/{Path}",
            shardName, fileSize / 1_000_000.0, _repoId, pathInRepo);

        const int maxRetries = 3;
        for (int attempt = 1; attempt <= maxRetries; attempt++)
        {
            try
            {
                // Step 1: Compute SHA-256 hash
                string sha256;
                await using (var fs = File.OpenRead(readyPath))
                {
                    var hash = await SHA256.HashDataAsync(fs, ct);
                    sha256 = Convert.ToHexStringLower(hash);
                }

                // Step 2: Request LFS presigned upload URL
                string lfsOid = sha256;
                var lfsPayload = new
                {
                    operation = "upload",
                    transfers = new[] { "basic" },
                    @ref = new { name = "main" },
                    objects = new[] { new { oid = lfsOid, size = fileSize } }
                };

                string lfsUrl = $"https://huggingface.co/{_repoId}.git/info/lfs/objects/batch";
                using var lfsRequest = new HttpRequestMessage(HttpMethod.Post, lfsUrl);
                lfsRequest.Content = new StringContent(
                    JsonSerializer.Serialize(lfsPayload), Encoding.UTF8, "application/vnd.git-lfs+json");
                lfsRequest.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("application/vnd.git-lfs+json"));

                using var lfsResponse = await _http.SendAsync(lfsRequest, ct);
                string lfsBody = await lfsResponse.Content.ReadAsStringAsync(ct);

                if (!lfsResponse.IsSuccessStatusCode)
                {
                    _log.LogWarning("LFS batch request failed (HTTP {Status}): {Body}",
                        (int)lfsResponse.StatusCode, lfsBody);
                    throw new HttpRequestException($"LFS batch failed: {lfsResponse.StatusCode}");
                }

                // Parse upload URL from LFS response
                using var lfsDoc = JsonDocument.Parse(lfsBody);
                var obj = lfsDoc.RootElement.GetProperty("objects")[0];

                // Check if already uploaded (no "actions" means it exists)
                if (!obj.TryGetProperty("actions", out var actions))
                {
                    _log.LogInformation("Shard {Shard} already exists in LFS, skipping upload", shardName);
                }
                else
                {
                    string uploadUrl = actions.GetProperty("upload").GetProperty("href").GetString()!;
                    var uploadHeaders = actions.GetProperty("upload").TryGetProperty("header", out var hdrs)
                        ? hdrs : default;

                    // Step 3: Upload file to LFS storage
                    _log.LogInformation("Uploading {Shard} to LFS storage...", shardName);
                    await using var uploadStream = File.OpenRead(readyPath);
                    using var uploadContent = new StreamContent(uploadStream);
                    uploadContent.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");
                    uploadContent.Headers.ContentLength = fileSize;

                    using var uploadRequest = new HttpRequestMessage(HttpMethod.Put, uploadUrl);
                    uploadRequest.Content = uploadContent;

                    // Add any custom headers from LFS response
                    if (uploadHeaders.ValueKind == JsonValueKind.Object)
                    {
                        foreach (var hdr in uploadHeaders.EnumerateObject())
                            uploadRequest.Headers.TryAddWithoutValidation(hdr.Name, hdr.Value.GetString());
                    }

                    using var uploadResponse = await _http.SendAsync(uploadRequest, ct);
                    if (!uploadResponse.IsSuccessStatusCode)
                    {
                        string uploadBody = await uploadResponse.Content.ReadAsStringAsync(ct);
                        throw new HttpRequestException($"LFS upload failed ({uploadResponse.StatusCode}): {uploadBody}");
                    }
                }

                // Step 4: Create commit with LFS pointer
                string lfsPointer = $"version https://git-lfs.github.com/spec/v1\n" +
                    $"oid sha256:{sha256}\n" +
                    $"size {fileSize}\n";

                string commitUrl = $"https://huggingface.co/api/datasets/{_repoId}/commit/main";
                var commitPayload = new
                {
                    summary = $"Add {shardName}",
                    lfsFiles = new[]
                    {
                        new
                        {
                            path = pathInRepo,
                            algo = "sha256",
                            oid = sha256,
                            size = fileSize
                        }
                    },
                    deletedFiles = Array.Empty<string>()
                };

                using var commitContent = new StringContent(
                    JsonSerializer.Serialize(commitPayload), Encoding.UTF8, "application/json");
                using var commitResponse = await _http.PostAsync(commitUrl, commitContent, ct);

                if (commitResponse.IsSuccessStatusCode)
                {
                    _log.LogInformation("Uploaded {Shard} successfully ({Size:F0} MB)",
                        shardName, fileSize / 1_000_000.0);
                    return;
                }

                string commitBody = await commitResponse.Content.ReadAsStringAsync(ct);
                _log.LogWarning("Commit failed (HTTP {Status}): {Body}",
                    (int)commitResponse.StatusCode, commitBody);
                throw new HttpRequestException($"Commit failed: {commitResponse.StatusCode}");
            }
            catch (Exception ex) when (attempt < maxRetries)
            {
                int delay = attempt * 15;
                _log.LogWarning(ex, "Upload {Shard} failed (attempt {Attempt}/{Max}), retrying in {Delay}s",
                    shardName, attempt, maxRetries, delay);
                await Task.Delay(delay * 1000, ct);
            }
        }

        _log.LogError("Failed to upload {Shard} after {Max} attempts. File preserved at {Path}",
            shardName, maxRetries, readyPath);
        throw new IOException($"Failed to upload {shardName} after {maxRetries} attempts");
    }
}
