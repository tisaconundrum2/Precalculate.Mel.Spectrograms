using Microsoft.Extensions.Options;
using Precalculate.Mel.Spectrograms.Contracts;
using Precalculate.Mel.Spectrograms.Models;
using Precalculate.Mel.Spectrograms.Services;

namespace Precalculate.Mel.Spectrograms.Workers;

public sealed class MelShardWorker(
    IMelSource source,
    IShardPublisher publisher,
    IOptions<ShardOptions> shardOptions,
    IHostApplicationLifetime lifetime,
    ILogger<MelShardWorker> log) : BackgroundService
{
    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        try
        {
            await RunAsync(ct);
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            // Normal host shutdown.
        }
        catch (Exception ex)
        {
            Environment.ExitCode = 1;
            log.LogError(ex, "Mel sharding failed");
            throw;
        }
        finally
        {
            // This is a finite batch job, not a perpetual service.
            lifetime.StopApplication();
        }
    }

    private async Task RunAsync(CancellationToken ct)
    {
        var options = shardOptions.Value;

        if (options.SamplesPerShard < 1 || options.MaxMelBytes < 1)
            throw new ArgumentException("Shard limits must be positive.");

        string spool = options.SpoolDirectory;
        Directory.CreateDirectory(spool);

        if (Directory.EnumerateFileSystemEntries(spool).Any())
            throw new IOException($"Recover nonempty spool first: {spool}");

        // Resume: check Hub for existing shards and skip forward
        int index = 0;
        int samplesToSkip = 0;
        string? hubRepo = options.HubRepo;
        string? hubPrefix = options.HubPrefix;
        string? hfToken = Environment.GetEnvironmentVariable("HF_TOKEN");

        if (!string.IsNullOrEmpty(hubRepo) && !string.IsNullOrEmpty(hfToken))
        {
            int existingShards = await CountExistingShardsAsync(hubRepo, hubPrefix, hfToken, ct);
            if (existingShards > 0)
            {
                index = existingShards;
                samplesToSkip = existingShards * options.SamplesPerShard;
                log.LogInformation(
                    "Resuming: found {Shards} existing shards, skipping {Samples} samples, starting at shard_{Index:D4}",
                    existingShards, samplesToSkip, index);
            }
        }

        MelShardWriter? writer = null;
        string? partial = null;

        async Task SealAsync()
        {
            if (writer is null || partial is null)
                return;

            string name = $"shard_{index:D4}.tar";
            string ready = Path.Combine(spool, name);
            int count = writer.Count;

            var closing = writer;
            writer = null;
            closing.Dispose();

            File.Move(partial, ready);
            partial = null;

            log.LogInformation(
                "Publishing {Shard}: {Count} samples, {Bytes} bytes",
                name, count, new FileInfo(ready).Length);

            await publisher.PublishAsync(ready, name, ct);

            // A local publisher may already have moved the file.
            File.Delete(ready);
            index++;
        }

        try
        {
            int skipped = 0;
            await foreach (var sample in source.ReadAsync(ct)
                .WithCancellation(ct))
            {
                // Skip samples already published in previous runs
                if (skipped < samplesToSkip)
                {
                    skipped++;
                    if (skipped % 5000 == 0)
                        log.LogInformation("Skipping: {Skipped}/{Total}",
                            skipped, samplesToSkip);
                    continue;
                }

                ct.ThrowIfCancellationRequested();
                long bytes = (long)sample.Values.Length * sizeof(float);

                if (bytes > options.MaxMelBytes)
                    throw new InvalidDataException(
                        $"Sample exceeds shard byte limit: {sample.Id}");

                if (writer is not null &&
                    bytes > options.MaxMelBytes - writer.MelBytes)
                {
                    await SealAsync();
                }

                if (writer is null)
                {
                    partial = Path.Combine(
                        spool, $"shard_{index:D4}.partial");
                    writer = new MelShardWriter(partial);
                }

                writer.Append(sample);

                if (writer.Count == options.SamplesPerShard)
                    await SealAsync();
            }

            await SealAsync(); // Publish the final short shard.
        }
        finally
        {
            try
            {
                writer?.Dispose();
            }
            finally
            {
                if (partial is not null)
                    File.Delete(partial);
            }
        }
    }

    private static async Task<int> CountExistingShardsAsync(
        string repoId, string? prefix, string token, CancellationToken ct)
    {
        using var http = new HttpClient();
        http.DefaultRequestHeaders.Authorization =
            new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", token);

        string url = $"https://huggingface.co/api/datasets/{repoId}";
        try
        {
            var response = await http.GetStringAsync(url, ct);
            // Count shard_NNNN.tar files in the siblings list
            string pattern = string.IsNullOrEmpty(prefix) ? "shard_" : $"{prefix}/shard_";
            int count = 0;
            int idx = 0;
            while ((idx = response.IndexOf(pattern, idx, StringComparison.Ordinal)) >= 0)
            {
                // Verify it's a .tar file reference
                int endIdx = response.IndexOf('"', idx);
                if (endIdx > idx && response[..endIdx].EndsWith(".tar"))
                    count++;
                idx += pattern.Length;
            }
            return count;
        }
        catch
        {
            return 0;
        }
    }
}
