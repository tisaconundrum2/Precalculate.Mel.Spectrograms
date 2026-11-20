namespace FlacDecoder;

/// <summary>
/// Reads individual bits from a byte buffer. Used for FLAC subframe decoding.
/// </summary>
internal sealed class BitReader
{
    private readonly byte[] _data;
    private int _bytePos;
    private int _bitPos; // 0-7, bits remaining in current byte (MSB first)

    public BitReader(byte[] data, int offset = 0)
    {
        _data = data;
        _bytePos = offset;
        _bitPos = 0;
    }

    public int BytePosition => _bytePos;
    public int BitPosition => _bytePos * 8 + _bitPos;

    public uint ReadBits(int count)
    {
        uint result = 0;
        for (int i = 0; i < count; i++)
        {
            result <<= 1;
            if ((_data[_bytePos] & (0x80 >> _bitPos)) != 0)
                result |= 1;
            _bitPos++;
            if (_bitPos == 8)
            {
                _bitPos = 0;
                _bytePos++;
            }
        }
        return result;
    }

    public int ReadSignedBits(int count)
    {
        uint val = ReadBits(count);
        // Sign extend
        if ((val & (1u << (count - 1))) != 0)
            val |= uint.MaxValue << count;
        return (int)val;
    }

    /// <summary>
    /// Read a unary-coded value (count of 0 bits before a 1 bit).
    /// </summary>
    public int ReadUnary()
    {
        int count = 0;
        while (ReadBits(1) == 0)
            count++;
        return count;
    }

    /// <summary>
    /// Read a UTF-8-like coded value (used in FLAC frame headers).
    /// </summary>
    public long ReadUtf8()
    {
        uint first = ReadBits(8);
        if (first < 0x80) return first;

        int extraBytes;
        long result;
        if ((first & 0xE0) == 0xC0) { extraBytes = 1; result = first & 0x1F; }
        else if ((first & 0xF0) == 0xE0) { extraBytes = 2; result = first & 0x0F; }
        else if ((first & 0xF8) == 0xF0) { extraBytes = 3; result = first & 0x07; }
        else if ((first & 0xFC) == 0xF8) { extraBytes = 4; result = first & 0x03; }
        else if ((first & 0xFE) == 0xFC) { extraBytes = 5; result = first & 0x01; }
        else if (first == 0xFE) { extraBytes = 6; result = 0; }
        else throw new InvalidDataException($"Invalid UTF-8 lead byte: 0x{first:X2}");

        for (int i = 0; i < extraBytes; i++)
        {
            uint b = ReadBits(8);
            if ((b & 0xC0) != 0x80)
                throw new InvalidDataException("Invalid UTF-8 continuation byte");
            result = (result << 6) | (b & 0x3F);
        }
        return result;
    }

    public void AlignToByte()
    {
        if (_bitPos != 0)
        {
            _bitPos = 0;
            _bytePos++;
        }
    }

    public void Skip(int bits)
    {
        _bitPos += bits;
        _bytePos += _bitPos / 8;
        _bitPos %= 8;
    }
}
