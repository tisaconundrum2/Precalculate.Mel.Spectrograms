---
title: Precalculate Mel Spectrograms
emoji: 🎵
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Precalculate Mel Spectrograms

Batch worker that reads audio datasets, computes mel spectrograms, and writes sharded `.tar` archives suitable for streaming training.

## HF Jobs usage

```bash
hf jobs run \
  --flavor t4-small \
  --timeout 6h \
  --attempts 2 \
  --name cache-1750h-mounted-v2 \
  -s HF_TOKEN \
  --detach \
  ghcr.io/tisaconundrum2/grassmann:latest \
  -- dotnet Precalculate.Mel.Spectrograms.dll
```

## Mounted datasets

| Name | Repo | Mount path | Mode |
|------|------|------------|------|
| grassmann-mel-cache | `tisaconundrum/grassmann-mel-cache` | `/data/tisaconundrum/grassmann-mel-cache` | read-write |
| librispeech | `openslr/librispeech_asr` | `/data/openslr/librispeech_asr` | read-only |
| voxpopuli | `facebook/voxpopuli` | `/data/facebook/voxpopuli` | read-only |
| peoples-speech | `MLCommons/peoples_speech` | `/data/MLCommons/peoples_speech` | read-only |

## Configuration

All settings are in `appsettings.json` and can be overridden via environment variables:

| Setting | Default | Description |
|---------|---------|-------------|
| `Shard__SpoolDirectory` | `/tmp/mel-spool` | Temp directory for in-progress shards |
| `Shard__OutputDirectory` | `output` | Local publish destination |
| `Shard__SamplesPerShard` | `5000` | Max samples before rotating a shard |
| `Shard__MaxMelBytes` | `2000000000` | Max mel payload bytes per shard (~2 GB) |
| `Mounts__BasePath` | `/data` | HF dataset mount root |

## Shard format

Each shard is a USTAR `.tar` containing pairs of entries per sample:

- `000000.f32` — little-endian float32 mel values, shape `[mel_bins, frames]`
- `000000.json` — metadata: `{ version, id, shape, dtype, token_ids }`

## Building the container

```bash
docker build -t grassmann-mel-cache .
```

## Local testing

```bash
docker run --rm \
  -v $(pwd)/output:/app/output \
  grassmann-mel-cache
```
