FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

COPY Precalculate.Mel.Spectrograms/Precalculate.Mel.Spectrograms.csproj \
     Precalculate.Mel.Spectrograms/
RUN dotnet restore Precalculate.Mel.Spectrograms/Precalculate.Mel.Spectrograms.csproj \
    -r linux-x64

COPY Precalculate.Mel.Spectrograms/ Precalculate.Mel.Spectrograms/
RUN dotnet publish Precalculate.Mel.Spectrograms/Precalculate.Mel.Spectrograms.csproj \
    -c Release \
    -r linux-x64 \
    --self-contained false \
    -o /app \
    --no-restore

# CUDA runtime with .NET runtime copied from official image
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
WORKDIR /app

# Copy .NET runtime from the official image
COPY --from=mcr.microsoft.com/dotnet/runtime:10.0 /usr/share/dotnet /usr/share/dotnet
RUN ln -s /usr/share/dotnet/dotnet /usr/bin/dotnet

# Install ffmpeg + python3 for PyTorch (libtorch sidecar)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libicu70 libgomp1 python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.1 — TorchSharp binds to its native libs
RUN pip3 install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Point linker at PyTorch's native libs so TorchSharp finds libtorch
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH

# HF Jobs mount datasets at /data; spool is ephemeral
RUN mkdir -p /tmp/mel-spool /data

COPY --from=build /app .

ENTRYPOINT ["dotnet", "Precalculate.Mel.Spectrograms.dll"]
