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

// Default to local filesystem publisher. Replace with a Hub publisher for HF jobs.
var outputDir = builder.Configuration["Shard:OutputDirectory"] ?? "output";
builder.Services.AddSingleton<IShardPublisher>(
    new DirectoryShardPublisher(outputDir));

builder.Services.AddHostedService<MelShardWorker>();

var host = builder.Build();
host.Run();
