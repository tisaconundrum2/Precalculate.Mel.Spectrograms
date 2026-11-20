using Precalculate.Mel.Spectrograms.Models;

namespace Precalculate.Mel.Spectrograms.Contracts;

public interface IMelSource
{
    /// <summary>
    /// Values are row-major: [mel bin, frame].
    /// Buffers must remain valid until the next MoveNextAsync call.
    /// </summary>
    IAsyncEnumerable<MelSample> ReadAsync(CancellationToken ct);
}
