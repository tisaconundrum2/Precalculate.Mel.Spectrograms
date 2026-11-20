using Precalculate.Mel.Spectrograms.Contracts;
using Precalculate.Mel.Spectrograms.Models;
using Precalculate.Mel.Spectrograms.Services;
using Precalculate.Mel.Spectrograms.Workers;

var builder = Host.CreateApplicationBuilder(args);

builder.Services.Configure<ShardOptions>(
    builder.Configuration.GetSection(ShardOptions.SectionName));
builder.Services.Configure<MountOptions>(
    builder.Configuration.GetSection(MountOptions.SectionName));
builder.Services.Configure<RecipeOptions>(
    builder.Configuration.GetSection(RecipeOptions.SectionName));

builder.Services.AddSingleton<IMelSource, HuggingFaceMelSource>();

// Use HF Hub publisher if HF_TOKEN and repo are configured, otherwise local filesystem
var hfToken = Environment.GetEnvironmentVariable("HF_TOKEN");
var hubRepo = builder.Configuration["Shard:HubRepo"];
if (!string.IsNullOrEmpty(hfToken) && !string.IsNullOrEmpty(hubRepo))
{
    var hubPrefix = builder.Configuration["Shard:HubPrefix"] ?? "";
    builder.Services.AddSingleton<IShardPublisher>(sp =>
        new HubShardPublisher(hubRepo, hubPrefix, hfToken,
            sp.GetRequiredService<ILogger<HubShardPublisher>>()));
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
