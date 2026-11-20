FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

COPY FlacDecoder/FlacDecoder.csproj FlacDecoder/
COPY Precalculate.Mel.Spectrograms/Precalculate.Mel.Spectrograms.csproj \
     Precalculate.Mel.Spectrograms/
RUN dotnet restore Precalculate.Mel.Spectrograms/Precalculate.Mel.Spectrograms.csproj

COPY FlacDecoder/ FlacDecoder/
COPY Precalculate.Mel.Spectrograms/ Precalculate.Mel.Spectrograms/
RUN dotnet publish Precalculate.Mel.Spectrograms/Precalculate.Mel.Spectrograms.csproj \
    -c Release \
    -o /app \
    --no-restore

FROM mcr.microsoft.com/dotnet/runtime:10.0
WORKDIR /app

# Install huggingface-cli for shard uploads (proven LFS handling)
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip \
    && pip3 install --no-cache-dir --break-system-packages huggingface-hub[cli] \
    && apt-get remove -y python3-pip && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# HF Jobs mount datasets at /data; spool is ephemeral
RUN mkdir -p /tmp/mel-spool /data

COPY --from=build /app .

ENTRYPOINT ["dotnet", "Precalculate.Mel.Spectrograms.dll"]
