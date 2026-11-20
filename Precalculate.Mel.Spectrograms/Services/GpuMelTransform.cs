using TorchSharp;
using static TorchSharp.torch;

namespace Precalculate.Mel.Spectrograms.Services;

/// <summary>
/// GPU-accelerated mel spectrogram using TorchSharp (libtorch).
/// Processes batches of audio clips on CUDA, falling back to CPU if no GPU available.
/// Matches Python: MelSpectrogram(sr=16000, n_fft=400, hop=160, n_mels=80) → log → z-norm.
/// </summary>
public sealed class GpuMelTransform : IDisposable
{
    private readonly int _nFft;
    private readonly int _hopLength;
    private readonly int _nMels;
    private readonly int _sampleRate;
    private readonly Device _device;
    private readonly Tensor _window;
    private readonly Tensor _melFilterbank;

    public int MelBins => _nMels;
    public bool IsGpu => _device.type == DeviceType.CUDA;

    public GpuMelTransform(
        int sampleRate = 16_000,
        int nFft = 400,
        int hopLength = 160,
        int nMels = 80)
    {
        _sampleRate = sampleRate;
        _nFft = nFft;
        _hopLength = hopLength;
        _nMels = nMels;

        _device = torch.cuda.is_available() ? CUDA : CPU;

        // Pre-compute window and mel filterbank on device
        _window = torch.hann_window(_nFft, device: _device);
        _melFilterbank = BuildMelFilterbank(nMels, nFft, sampleRate).to(_device);
    }

    /// <summary>
    /// Compute mel spectrogram for a single audio clip.
    /// Returns (values, frames) where values is row-major [mel_bins, frames].
    /// </summary>
    public (float[] Values, int Frames) Compute(ReadOnlySpan<float> audio)
    {
        using var waveform = torch.tensor(audio.ToArray(), dtype: ScalarType.Float32, device: _device);
        using var mel = ComputeMel(waveform);

        int frames = (int)mel.shape[1];
        var values = new float[_nMels * frames];
        mel.to(CPU).data<float>().CopyTo(values);
        return (values, frames);
    }

    /// <summary>
    /// Batch compute mel spectrograms for multiple audio clips.
    /// Pads to longest clip, computes all at once on GPU, returns individual results.
    /// </summary>
    public (float[] Values, int Frames)[] ComputeBatch(float[][] audioClips)
    {
        if (audioClips.Length == 0) return [];
        if (audioClips.Length == 1) return [Compute(audioClips[0])];

        // Pad all clips to max length
        int maxLen = 0;
        var frameCounts = new int[audioClips.Length];
        foreach (var clip in audioClips)
            if (clip.Length > maxLen) maxLen = clip.Length;

        // Create padded batch tensor
        using var batch = torch.zeros([audioClips.Length, maxLen],
            dtype: ScalarType.Float32, device: _device);

        for (int i = 0; i < audioClips.Length; i++)
        {
            using var clip = torch.tensor(audioClips[i], dtype: ScalarType.Float32, device: _device);
            batch[i, ..(int)clip.shape[0]] = clip;
            frameCounts[i] = 1 + (audioClips[i].Length - _nFft) / _hopLength;
        }

        // Batch STFT
        using var stft = torch.stft(
            batch, _nFft,
            hop_length: _hopLength,
            window: _window,
            return_complex: true);

        // Power spectrum: |STFT|^2
        using var power = stft.abs().pow(2); // [batch, n_fft/2+1, frames]

        // Apply mel filterbank: [n_mels, n_fft/2+1] @ [batch, n_fft/2+1, frames]
        // → [batch, n_mels, frames]
        using var melSpec = torch.matmul(_melFilterbank, power);

        // Log
        using var logMel = melSpec.clamp_min(1e-9f).log();

        // Move to CPU and extract individual results
        using var cpuMel = logMel.to(CPU);

        var results = new (float[] Values, int Frames)[audioClips.Length];
        for (int i = 0; i < audioClips.Length; i++)
        {
            int frames = frameCounts[i];
            if (frames <= 0) continue;

            // Extract [n_mels, actual_frames] (strip padding)
            using var singleMel = cpuMel[i, .., ..frames];

            // Z-normalize per sample
            using var mean = singleMel.mean();
            using var std = singleMel.std().clamp_min(1e-9f);
            using var normed = (singleMel - mean) / std;

            var values = new float[_nMels * frames];
            normed.data<float>().CopyTo(values);
            results[i] = (values, frames);
        }
        return results;
    }

    private Tensor ComputeMel(Tensor waveform)
    {
        // STFT
        using var stft = torch.stft(
            waveform.unsqueeze(0), _nFft,
            hop_length: _hopLength,
            window: _window,
            return_complex: true);

        // Power spectrum
        using var power = stft.abs().pow(2).squeeze(0); // [n_fft/2+1, frames]

        // Mel filterbank
        using var melSpec = torch.matmul(_melFilterbank, power); // [n_mels, frames]

        // Log
        using var logMel = melSpec.clamp_min(1e-9f).log();

        // Z-normalize
        using var mean = logMel.mean();
        using var std = logMel.std().clamp_min(1e-9f);
        return (logMel - mean) / std;
    }

    private static Tensor BuildMelFilterbank(int nMels, int nFft, int sampleRate)
    {
        int nBins = nFft / 2 + 1;
        float fMin = 0f, fMax = sampleRate / 2f;
        float melMin = 2595f * MathF.Log10(1f + fMin / 700f);
        float melMax = 2595f * MathF.Log10(1f + fMax / 700f);

        var melPoints = new float[nMels + 2];
        for (int i = 0; i < nMels + 2; i++)
            melPoints[i] = melMin + i * (melMax - melMin) / (nMels + 1);

        var hzPoints = new float[nMels + 2];
        for (int i = 0; i < nMels + 2; i++)
            hzPoints[i] = 700f * (MathF.Pow(10f, melPoints[i] / 2595f) - 1f);

        var binPoints = new int[nMels + 2];
        for (int i = 0; i < nMels + 2; i++)
            binPoints[i] = (int)MathF.Floor((nFft + 1) * hzPoints[i] / sampleRate);

        var fb = new float[nMels * nBins];
        for (int m = 0; m < nMels; m++)
        {
            int left = binPoints[m], center = binPoints[m + 1], right = binPoints[m + 2];
            for (int k = left; k < center; k++)
                if (center > left) fb[m * nBins + k] = (float)(k - left) / (center - left);
            for (int k = center; k < right; k++)
                if (right > center) fb[m * nBins + k] = (float)(right - k) / (right - center);
        }

        return torch.tensor(fb, [nMels, nBins], dtype: ScalarType.Float32);
    }

    public void Dispose()
    {
        _window.Dispose();
        _melFilterbank.Dispose();
    }
}
