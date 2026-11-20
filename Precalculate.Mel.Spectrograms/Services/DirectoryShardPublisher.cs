using Precalculate.Mel.Spectrograms.Contracts;

namespace Precalculate.Mel.Spectrograms.Services;

public sealed class DirectoryShardPublisher(string output) : IShardPublisher
{
    public Task PublishAsync(
        string readyPath,
        string shardName,
        CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        Directory.CreateDirectory(output);

        string destination = Path.Combine(output, shardName);
        File.Move(readyPath, destination, overwrite: false);
        return Task.CompletedTask;
    }
}
