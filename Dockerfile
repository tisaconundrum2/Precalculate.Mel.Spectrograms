FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

COPY Precalculate.Mel.Spectrograms/Precalculate.Mel.Spectrograms.csproj \
     Precalculate.Mel.Spectrograms/
RUN dotnet restore Precalculate.Mel.Spectrograms/Precalculate.Mel.Spectrograms.csproj

COPY Precalculate.Mel.Spectrograms/ Precalculate.Mel.Spectrograms/
RUN dotnet publish Precalculate.Mel.Spectrograms/Precalculate.Mel.Spectrograms.csproj \
    -c Release \
    -o /app \
    --no-restore

# CUDA runtime with .NET runtime copied from official image
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
WORKDIR /app

# Copy .NET runtime from the official image
COPY --from=mcr.microsoft.com/dotnet/runtime:10.0 /usr/share/dotnet /usr/share/dotnet
RUN ln -s /usr/share/dotnet/dotnet /usr/bin/dotnet

# Install ffmpeg + tools for libtorch download
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libicu70 libgomp1 wget unzip \
    && rm -rf /var/lib/apt/lists/*

# Install libtorch C++ distribution (TorchSharp native backend)
RUN wget -q "https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu121.zip" \
    -O /tmp/libtorch.zip \
    && unzip -q /tmp/libtorch.zip -d /opt \
    && rm /tmp/libtorch.zip

ENV LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH

# HF Jobs mount datasets at /data; spool is ephemeral
RUN mkdir -p /tmp/mel-spool /data

COPY --from=build /app .

ENTRYPOINT ["dotnet", "Precalculate.Mel.Spectrograms.dll"]
