using System.Numerics;

namespace Precalculate.Mel.Spectrograms.Services;

/// <summary>
/// Pure C# mel spectrogram: STFT → power spectrum → mel filterbank → log → normalize.
/// Matches the Python pipeline: torchaudio.transforms.MelSpectrogram with
/// sample_rate=16000, n_fft=400, hop_length=160, n_mels=80, then log(clamp(1e-9)) + z-norm.
/// </summary>
public sealed class MelTransform
{
    private readonly int _nFft;
    private readonly int _hopLength;
    private readonly int _nMels;
    private readonly int _sampleRate;
    private readonly float[] _window;
    private readonly float[,] _filterbank;
    private readonly int _fftSize; // next power of 2 >= _nFft

    public int MelBins => _nMels;

    public MelTransform(
        int sampleRate = 16_000,
        int nFft = 400,
        int hopLength = 160,
        int nMels = 80)
    {
        _sampleRate = sampleRate;
        _nFft = nFft;
        _hopLength = hopLength;
        _nMels = nMels;
        _fftSize = NextPowerOf2(nFft);
        _window = BuildHannWindow(nFft);
        _filterbank = BuildMelFilterbank(nMels, nFft, sampleRate);
    }

    /// <summary>
    /// Compute log-mel spectrogram from mono float32 PCM at the expected sample rate.
    /// Returns row-major [mel_bins, frames] as a flat float array + frame count.
    /// </summary>
    public (float[] Values, int Frames) Compute(ReadOnlySpan<float> audio)
    {
        int nFrames = 1 + (audio.Length - _nFft) / _hopLength;
        if (nFrames <= 0)
            throw new ArgumentException("Audio too short for a single frame.");

        int nBins = _nFft / 2 + 1;
        var mel = new float[_nMels * nFrames];

        // Reusable buffers
        var re = new float[_fftSize];
        var im = new float[_fftSize];

        for (int t = 0; t < nFrames; t++)
        {
            int offset = t * _hopLength;

            // Window + zero-pad
            for (int n = 0; n < _nFft; n++)
                re[n] = audio[offset + n] * _window[n];
            for (int n = _nFft; n < _fftSize; n++)
                re[n] = 0;
            Array.Clear(im, 0, _fftSize);

            Fft(re, im, _fftSize);

            // Apply mel filterbank + log
            for (int m = 0; m < _nMels; m++)
            {
                float sum = 0;
                for (int k = 0; k < nBins; k++)
                {
                    float power = re[k] * re[k] + im[k] * im[k];
                    sum += _filterbank[m, k] * power;
                }
                mel[m * nFrames + t] = MathF.Log(MathF.Max(sum, 1e-9f));
            }
        }

        // Z-normalize per sample
        float mean = 0;
        for (int i = 0; i < mel.Length; i++)
            mean += mel[i];
        mean /= mel.Length;

        float variance = 0;
        for (int i = 0; i < mel.Length; i++)
        {
            float diff = mel[i] - mean;
            variance += diff * diff;
        }
        float std = MathF.Max(MathF.Sqrt(variance / mel.Length), 1e-9f);

        for (int i = 0; i < mel.Length; i++)
            mel[i] = (mel[i] - mean) / std;

        return (mel, nFrames);
    }

    private static float[] BuildHannWindow(int size)
    {
        var w = new float[size];
        for (int i = 0; i < size; i++)
            w[i] = 0.5f * (1f - MathF.Cos(2f * MathF.PI * i / size));
        return w;
    }

    private static float[,] BuildMelFilterbank(int nMels, int nFft, int sampleRate)
    {
        int nBins = nFft / 2 + 1;
        float fMin = 0f;
        float fMax = sampleRate / 2f;
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

        var fb = new float[nMels, nBins];
        for (int m = 0; m < nMels; m++)
        {
            int left = binPoints[m];
            int center = binPoints[m + 1];
            int right = binPoints[m + 2];

            for (int k = left; k < center; k++)
            {
                if (center > left)
                    fb[m, k] = (float)(k - left) / (center - left);
            }
            for (int k = center; k < right; k++)
            {
                if (right > center)
                    fb[m, k] = (float)(right - k) / (right - center);
            }
        }
        return fb;
    }

    /// <summary>
    /// In-place radix-2 Cooley-Tukey FFT. Size must be a power of 2.
    /// </summary>
    private static void Fft(float[] re, float[] im, int n)
    {
        // Bit-reversal permutation
        for (int i = 1, j = 0; i < n; i++)
        {
            int bit = n >> 1;
            while ((j & bit) != 0)
            {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;

            if (i < j)
            {
                (re[i], re[j]) = (re[j], re[i]);
                (im[i], im[j]) = (im[j], im[i]);
            }
        }

        // Butterfly stages
        for (int len = 2; len <= n; len <<= 1)
        {
            float angle = -2f * MathF.PI / len;
            float wRe = MathF.Cos(angle);
            float wIm = MathF.Sin(angle);
            int half = len >> 1;

            for (int i = 0; i < n; i += len)
            {
                float curRe = 1f, curIm = 0f;
                for (int j = 0; j < half; j++)
                {
                    int idx = i + j + half;
                    float tRe = curRe * re[idx] - curIm * im[idx];
                    float tIm = curRe * im[idx] + curIm * re[idx];
                    re[idx] = re[i + j] - tRe;
                    im[idx] = im[i + j] - tIm;
                    re[i + j] += tRe;
                    im[i + j] += tIm;
                    float newCurRe = curRe * wRe - curIm * wIm;
                    curIm = curRe * wIm + curIm * wRe;
                    curRe = newCurRe;
                }
            }
        }
    }

    private static int NextPowerOf2(int v)
    {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        return v + 1;
    }
}
