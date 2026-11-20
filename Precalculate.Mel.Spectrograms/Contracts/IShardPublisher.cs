namespace Precalculate.Mel.Spectrograms.Contracts;

public interface IShardPublisher
{
    /// <summary>
    /// On success, the completed shard is durable at its destination.
    /// On failure, leave readyPath intact so publication can be retried.
    /// </summary>
    Task PublishAsync(
        string readyPath,
        string shardName,
        CancellationToken ct);
}
