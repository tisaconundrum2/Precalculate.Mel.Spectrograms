using Microsoft.Extensions.Options;
using Precalculate.Mel.Spectrograms.Contracts;
using Precalculate.Mel.Spectrograms.Models;
using Precalculate.Mel.Spectrograms.Services;
using Precalculate.Mel.Spectrograms.Workers;

var builder = Host.CreateApplicationBuilder(args);

builder.Services.Configure<ShardOptions>(
    builder.Configuration.GetSection(ShardOptions.SectionName));
builder.Services.Configure<MountOptions>(
    builder.Configuration.GetSection(MountOptions.SectionName));

// If RECIPE_NAME is set, use the named recipe from the Recipes dictionary
var recipeName = Environment.GetEnvironmentVariable("RECIPE_NAME");
if (!string.IsNullOrEmpty(recipeName))
{
    builder.Services.Configure<RecipeOptions>(
        builder.Configuration.GetSection($"Recipes:{recipeName}"));

    // Override shard hub prefix if the recipe specifies one
    var hubPrefixOverride = builder.Configuration[$"Recipes:{recipeName}:HubPrefix"];
    if (!string.IsNullOrEmpty(hubPrefixOverride))
    {
        builder.Services.PostConfigure<ShardOptions>(o => o.HubPrefix = hubPrefixOverride);
    }
}
else
{
    builder.Services.Configure<RecipeOptions>(
        builder.Configuration.GetSection(RecipeOptions.SectionName));
}

builder.Services.AddSingleton<IMelSource, HuggingFaceMelSource>();

// Use HF Hub publisher if HF_TOKEN and repo are configured, otherwise local filesystem
var hfToken = Environment.GetEnvironmentVariable("HF_TOKEN");
var hubRepo = builder.Configuration["Shard:HubRepo"];
if (!string.IsNullOrEmpty(hfToken) && !string.IsNullOrEmpty(hubRepo))
{
    builder.Services.AddSingleton<IShardPublisher>(sp =>
    {
        var opts = sp.GetRequiredService<IOptions<ShardOptions>>().Value;
        return new HubShardPublisher(hubRepo, opts.HubPrefix ?? "", hfToken,
            sp.GetRequiredService<ILogger<HubShardPublisher>>());
    });
}
else
{
    var outputDir = builder.Configuration["Shard:OutputDirectory"] ?? "output";
    builder.Services.AddSingleton<IShardPublisher>(
        new DirectoryShardPublisher(outputDir));
}

builder.Services.AddHostedService<MelShardWorker>();

var host = builder.Build();
host.Run();
