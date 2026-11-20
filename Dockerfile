FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

COPY Precalculate.Mel.Spectrograms/Precalculate.Mel.Spectrograms.csproj \
     Precalculate.Mel.Spectrograms/
RUN dotnet restore Precalculate.Mel.Spectrograms/Precalculate.Mel.Spectrograms.csproj \
    -p:BuildingInsideDocker=true

COPY Precalculate.Mel.Spectrograms/ Precalculate.Mel.Spectrograms/
RUN dotnet publish Precalculate.Mel.Spectrograms/Precalculate.Mel.Spectrograms.csproj \
    -c Release \
    -o /app \
    -p:BuildingInsideDocker=true \
    --no-restore

# CUDA runtime with .NET runtime copied from official image
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
WORKDIR /app

# Copy .NET runtime from the official image (avoids install script issues)
COPY --from=mcr.microsoft.com/dotnet/runtime:10.0 /usr/share/dotnet /usr/share/dotnet
RUN ln -s /usr/share/dotnet/dotnet /usr/bin/dotnet

# Install ffmpeg for audio decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libicu70 \
    && rm -rf /var/lib/apt/lists/*

# HF Jobs mount datasets at /data; spool is ephemeral
RUN mkdir -p /tmp/mel-spool /data

COPY --from=build /app .

ENTRYPOINT ["dotnet", "Precalculate.Mel.Spectrograms.dll"]
