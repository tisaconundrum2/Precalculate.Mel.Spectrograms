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

# CUDA runtime with .NET runtime installed on top
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
WORKDIR /app

# Install .NET 10 runtime + ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates ffmpeg \
    && wget -q https://dot.net/v1/dotnet-install.sh -O /tmp/dotnet-install.sh \
    && chmod +x /tmp/dotnet-install.sh \
    && /tmp/dotnet-install.sh --channel 10.0 --runtime dotnet --install-dir /usr/share/dotnet \
    && ln -s /usr/share/dotnet/dotnet /usr/bin/dotnet \
    && rm /tmp/dotnet-install.sh \
    && apt-get remove -y wget \
COPY --from=build /app .

ENTRYPOINT ["dotnet", "Precalculate.Mel.Spectrograms.dll"]
