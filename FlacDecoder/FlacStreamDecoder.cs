namespace FlacDecoder;

/// <summary>
/// Minimal managed FLAC decoder. Decodes FLAC streams to 16-bit PCM samples.
/// Supports mono and stereo, any sample rate, 16-bit depth.
/// No external dependencies — pure C#.
/// </summary>
public sealed class FlacStreamDecoder
{
    // STREAMINFO fields
    public int SampleRate { get; private set; }
    public int Channels { get; private set; }
    public int BitsPerSample { get; private set; }
    public long TotalSamples { get; private set; }
    public int MinBlockSize { get; private set; }
    public int MaxBlockSize { get; private set; }

    /// <summary>
    /// Decode a FLAC byte array to interleaved 16-bit PCM samples.
    /// Returns the raw sample values as short[].
    /// </summary>
    public short[] Decode(byte[] flacData)
    {
        int pos = 0;

        // Verify magic
        if (flacData.Length < 4 ||
            flacData[0] != (byte)'f' || flacData[1] != (byte)'L' ||
            flacData[2] != (byte)'a' || flacData[3] != (byte)'C')
            throw new InvalidDataException("Not a FLAC file (missing fLaC magic)");
        pos = 4;

        // Read metadata blocks
        bool lastBlock = false;
        while (!lastBlock && pos < flacData.Length)
        {
            lastBlock = (flacData[pos] & 0x80) != 0;
            int blockType = flacData[pos] & 0x7F;
            int blockLength = (flacData[pos + 1] << 16) | (flacData[pos + 2] << 8) | flacData[pos + 3];
            pos += 4;

            if (blockType == 0) // STREAMINFO
                ReadStreamInfo(flacData, pos);

            pos += blockLength;
        }

        if (SampleRate == 0)
            throw new InvalidDataException("No STREAMINFO block found");

        // Decode audio frames
        var allSamples = new List<short>((int)(TotalSamples * Channels));

        while (pos < flacData.Length - 2) // Need at least sync code
        {
            // Look for frame sync: 0xFFF8 or 0xFFF9 (14 bits of 1s + reserved + blocking strategy)
            if (flacData[pos] != 0xFF || (flacData[pos + 1] & 0xFC) != 0xF8)
            {
                pos++;
                continue;
            }

            int frameStart = pos;
            try
            {
                var frameSamples = DecodeFrame(flacData, ref pos);
                allSamples.AddRange(frameSamples);
            }
            catch (IndexOutOfRangeException)
            {
                // Hit end of data mid-frame — we're done
                break;
            }
            catch (Exception ex)
            {
                // Skip corrupted frame, try to find next sync
                System.Diagnostics.Debug.WriteLine($"Frame decode error at {frameStart}: {ex.Message}");
                pos = frameStart + 1;
            }
        }

        return allSamples.ToArray();
    }

    /// <summary>
    /// Decode FLAC to mono float32 PCM normalized to [-1, 1].
    /// If stereo, downmixes to mono.
    /// </summary>
    public float[] DecodeToFloat32Mono(byte[] flacData)
    {
        short[] pcm = Decode(flacData);

        if (Channels == 1)
        {
            var result = new float[pcm.Length];
            for (int i = 0; i < pcm.Length; i++)
                result[i] = pcm[i] / 32768f;
            return result;
        }

        // Downmix to mono
        int sampleCount = pcm.Length / Channels;
        var mono = new float[sampleCount];
        for (int i = 0; i < sampleCount; i++)
        {
            float sum = 0;
            for (int ch = 0; ch < Channels; ch++)
                sum += pcm[i * Channels + ch];
            mono[i] = sum / Channels / 32768f;
        }
        return mono;
    }

    private void ReadStreamInfo(byte[] data, int offset)
    {
        MinBlockSize = (data[offset] << 8) | data[offset + 1];
        MaxBlockSize = (data[offset + 2] << 8) | data[offset + 3];
        // min/max frame size: 3 bytes each (skip)
        // sample rate: 20 bits, channels: 3 bits, bps: 5 bits, total samples: 36 bits
        int sr = (data[offset + 10] << 12) | (data[offset + 11] << 4) | (data[offset + 12] >> 4);
        int ch = ((data[offset + 12] >> 1) & 0x07) + 1;
        int bps = (((data[offset + 12] & 0x01) << 4) | (data[offset + 13] >> 4)) + 1;
        long total = ((long)(data[offset + 13] & 0x0F) << 32) |
                     ((long)data[offset + 14] << 24) |
                     ((long)data[offset + 15] << 16) |
                     ((long)data[offset + 16] << 8) |
                     data[offset + 17];

        SampleRate = sr;
        Channels = ch;
        BitsPerSample = bps;
        TotalSamples = total;
    }

    private short[] DecodeFrame(byte[] data, ref int pos)
    {
        var reader = new BitReader(data, pos);

        // Frame header
        uint sync = reader.ReadBits(14); // Should be 0x3FFE
        if (sync != 0x3FFE)
            throw new InvalidDataException("Bad frame sync");

        reader.ReadBits(1); // Reserved
        int blockingStrategy = (int)reader.ReadBits(1);

        int blockSizeCode = (int)reader.ReadBits(4);
        int sampleRateCode = (int)reader.ReadBits(4);
        int channelAssignment = (int)reader.ReadBits(4);
        int sampleSizeCode = (int)reader.ReadBits(3);
        reader.ReadBits(1); // Reserved

        // Frame/sample number (UTF-8 coded)
        reader.ReadUtf8();

        // Block size
        int blockSize = blockSizeCode switch
        {
            0 => throw new InvalidDataException("Reserved block size"),
            1 => 192,
            2 => 576,
            3 => 1152,
            4 => 2304,
            5 => 4608,
            6 => (int)reader.ReadBits(8) + 1,
            7 => (int)reader.ReadBits(16) + 1,
            >= 8 and <= 15 => 256 << (blockSizeCode - 8),
            _ => throw new InvalidDataException($"Invalid block size code: {blockSizeCode}")
        };

        // Sample rate (from header or override)
        if (sampleRateCode == 12) reader.ReadBits(8);
        else if (sampleRateCode == 13) reader.ReadBits(16);
        else if (sampleRateCode == 14) reader.ReadBits(16);

        // Bits per sample from frame header
        int bps = sampleSizeCode switch
        {
            0 => BitsPerSample, // from STREAMINFO
            1 => 8,
            2 => 12,
            3 => throw new InvalidDataException("Reserved sample size"),
            4 => 16,
            5 => 20,
            6 => 24,
            7 => 32,
            _ => throw new InvalidDataException($"Invalid sample size code: {sampleSizeCode}")
        };

        // Frame header CRC-8
        reader.ReadBits(8);

        // Determine actual channel count and decode mode
        int numChannels;
        bool isMidSide = false, isLeftSide = false, isRightSide = false;

        if (channelAssignment < 8)
        {
            numChannels = channelAssignment + 1;
        }
        else if (channelAssignment == 8) { numChannels = 2; isLeftSide = true; }
        else if (channelAssignment == 9) { numChannels = 2; isRightSide = true; }
        else if (channelAssignment == 10) { numChannels = 2; isMidSide = true; }
        else throw new InvalidDataException($"Invalid channel assignment: {channelAssignment}");

        // Decode subframes
        var channels = new int[numChannels][];
        for (int ch = 0; ch < numChannels; ch++)
        {
            int subframeBps = bps;
            // Side channel gets +1 bit
            if ((isLeftSide && ch == 1) || (isRightSide && ch == 0) || (isMidSide && ch == 1))
                subframeBps++;

            channels[ch] = DecodeSubframe(reader, blockSize, subframeBps);
        }

        // Apply stereo decorrelation
        if (isLeftSide)
        {
            // ch0 = left, ch1 = left - right → right = left - ch1
            for (int i = 0; i < blockSize; i++)
                channels[1][i] = channels[0][i] - channels[1][i];
        }
        else if (isRightSide)
        {
            // ch0 = left - right, ch1 = right → left = ch0 + right
            for (int i = 0; i < blockSize; i++)
                channels[0][i] = channels[0][i] + channels[1][i];
        }
        else if (isMidSide)
        {
            // ch0 = mid = (left + right) / 2, ch1 = side = left - right
            for (int i = 0; i < blockSize; i++)
            {
                int mid = channels[0][i];
                int side = channels[1][i];
                mid = (mid << 1) | (side & 1); // Restore full precision
                channels[0][i] = (mid + side) >> 1;  // left
                channels[1][i] = (mid - side) >> 1;  // right
            }
        }

        // Align to byte boundary and skip frame footer CRC-16
        reader.AlignToByte();
        reader.ReadBits(16); // CRC-16

        // BytePosition is absolute from the start of the data array
        pos = reader.BytePosition;

        // Interleave and clamp to 16-bit
        var output = new short[blockSize * numChannels];
        for (int i = 0; i < blockSize; i++)
        {
            for (int ch = 0; ch < numChannels; ch++)
            {
                int sample = channels[ch][i];
                // Shift to 16-bit if needed
                if (bps < 16) sample <<= (16 - bps);
                else if (bps > 16) sample >>= (bps - 16);
                output[i * numChannels + ch] = (short)Math.Clamp(sample, short.MinValue, short.MaxValue);
            }
        }

        return output;
    }

    private static int[] DecodeSubframe(BitReader reader, int blockSize, int bps)
    {
        // Subframe header
        reader.ReadBits(1); // Zero padding

        int type = (int)reader.ReadBits(6);
        bool hasWastedBits = reader.ReadBits(1) == 1;
        int wastedBits = 0;
        if (hasWastedBits)
        {
            wastedBits = reader.ReadUnary() + 1;
            bps -= wastedBits;
        }

        int[] samples;

        if (type == 0)
        {
            // CONSTANT
            int value = reader.ReadSignedBits(bps);
            samples = new int[blockSize];
            Array.Fill(samples, value);
        }
        else if (type == 1)
        {
            // VERBATIM
            samples = new int[blockSize];
            for (int i = 0; i < blockSize; i++)
                samples[i] = reader.ReadSignedBits(bps);
        }
        else if (type >= 8 && type <= 12)
        {
            // FIXED predictor, order = type - 8
            int order = type - 8;
            samples = DecodeFixed(reader, blockSize, bps, order);
        }
        else if (type >= 32 && type <= 63)
        {
            // LPC predictor, order = type - 31
            int order = type - 31;
            samples = DecodeLpc(reader, blockSize, bps, order);
        }
        else
        {
            throw new InvalidDataException($"Reserved subframe type: {type}");
        }

        // Restore wasted bits
        if (wastedBits > 0)
        {
            for (int i = 0; i < blockSize; i++)
                samples[i] <<= wastedBits;
        }

        return samples;
    }

    private static int[] DecodeFixed(BitReader reader, int blockSize, int bps, int order)
    {
        var samples = new int[blockSize];

        // Warm-up samples
        for (int i = 0; i < order; i++)
            samples[i] = reader.ReadSignedBits(bps);

        // Read residual
        int[] residual = DecodeResidual(reader, blockSize, order);

        // Apply fixed predictor
        for (int i = order; i < blockSize; i++)
        {
            int prediction = order switch
            {
                0 => 0,
                1 => samples[i - 1],
                2 => 2 * samples[i - 1] - samples[i - 2],
                3 => 3 * samples[i - 1] - 3 * samples[i - 2] + samples[i - 3],
                4 => 4 * samples[i - 1] - 6 * samples[i - 2] + 4 * samples[i - 3] - samples[i - 4],
                _ => throw new InvalidDataException($"Invalid fixed order: {order}")
            };
            samples[i] = prediction + residual[i - order];
        }

        return samples;
    }

    private static int[] DecodeLpc(BitReader reader, int blockSize, int bps, int order)
    {
        var samples = new int[blockSize];

        // Warm-up samples
        for (int i = 0; i < order; i++)
            samples[i] = reader.ReadSignedBits(bps);

        // LPC precision
        int precision = (int)reader.ReadBits(4) + 1;
        int shift = reader.ReadSignedBits(5);

        // LPC coefficients
        var coeffs = new int[order];
        for (int i = 0; i < order; i++)
            coeffs[i] = reader.ReadSignedBits(precision);

        // Read residual
        int[] residual = DecodeResidual(reader, blockSize, order);

        // Apply LPC predictor
        for (int i = order; i < blockSize; i++)
        {
            long prediction = 0;
            for (int j = 0; j < order; j++)
                prediction += (long)coeffs[j] * samples[i - 1 - j];
            samples[i] = (int)(prediction >> shift) + residual[i - order];
        }

        return samples;
    }

    private static int[] DecodeResidual(BitReader reader, int blockSize, int predictorOrder)
    {
        int codingMethod = (int)reader.ReadBits(2);
        int partitionOrder = (int)reader.ReadBits(4);
        int numPartitions = 1 << partitionOrder;

        int riceBits = codingMethod == 0 ? 4 : 5; // Rice or Rice2
        int escapeCode = codingMethod == 0 ? 15 : 31;

        int residualCount = blockSize - predictorOrder;
        var residual = new int[residualCount];

        int sampleIdx = 0;
        for (int partition = 0; partition < numPartitions; partition++)
        {
            int samplesInPartition = partition == 0
                ? (blockSize / numPartitions) - predictorOrder
                : blockSize / numPartitions;

            int riceParam = (int)reader.ReadBits(riceBits);

            if (riceParam == escapeCode)
            {
                // Escape: raw bits
                int rawBits = (int)reader.ReadBits(5);
                for (int i = 0; i < samplesInPartition; i++)
                    residual[sampleIdx++] = reader.ReadSignedBits(rawBits);
            }
            else
            {
                // Rice-coded
                for (int i = 0; i < samplesInPartition; i++)
                {
                    int quotient = reader.ReadUnary();
                    int remainder = riceParam > 0 ? (int)reader.ReadBits(riceParam) : 0;
                    int value = (quotient << riceParam) | remainder;
                    // Zigzag decode
                    residual[sampleIdx++] = (value & 1) != 0 ? -(value >> 1) - 1 : value >> 1;
                }
            }
        }

        return residual;
    }
}
