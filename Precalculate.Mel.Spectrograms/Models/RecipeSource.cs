namespace Precalculate.Mel.Spectrograms.Models;

public sealed class RecipeSource
{
    public required string Name { get; init; }
    public required string Split { get; init; }
    public string? Config { get; init; }
    public string? AudioColumn { get; init; }
    public string? TextColumn { get; init; }
    public int? MaxSamples { get; init; }
}

public sealed class RecipeOptions
{
    public const string SectionName = "Recipe";

    public required string Name { get; init; }
    public List<RecipeSource> Sources { get; init; } = [];
}
