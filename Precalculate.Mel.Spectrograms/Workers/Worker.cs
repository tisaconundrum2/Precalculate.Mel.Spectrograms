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

        int index = 0;
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
            await foreach (var sample in source.ReadAsync(ct)
                .WithCancellation(ct))
            {
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
}
