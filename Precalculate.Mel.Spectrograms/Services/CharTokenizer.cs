namespace Precalculate.Mel.Spectrograms.Services;

/// <summary>
/// Character-level tokenizer matching the Python CharTokenizer:
/// Index 0 = blank (CTC), 1 = space, 2-27 = a-z, 28 = apostrophe.
/// </summary>
public sealed class CharTokenizer
{
    // '\0 abcdefghijklmnopqrstuvwxyz\'' — 29 characters
    private const string Vocab = "\0 abcdefghijklmnopqrstuvwxyz'";

    private readonly Dictionary<char, int> _charToId;

    public int VocabSize => Vocab.Length; // 29
    public int BlankId => 0;

    public CharTokenizer()
    {
        _charToId = new Dictionary<char, int>(Vocab.Length);
        for (int i = 0; i < Vocab.Length; i++)
            _charToId[Vocab[i]] = i;
    }

    /// <summary>
    /// Encode normalized text to token IDs.
    /// Unknown characters are silently skipped.
    /// </summary>
    public int[] Encode(string text)
    {
        var normalized = Normalize(text);
        var tokens = new List<int>(normalized.Length);
        foreach (char c in normalized)
        {
            if (_charToId.TryGetValue(c, out int id))
                tokens.Add(id);
        }
        return tokens.ToArray();
    }

    /// <summary>
    /// Normalize transcript to lowercase ASCII: letters, spaces, apostrophes only.
    /// Matches Python normalize_transcript().
    /// </summary>
    public static string Normalize(string text)
    {
        var sb = new System.Text.StringBuilder(text.Length);
        bool lastWasSpace = false;

        foreach (char c in text.ToLowerInvariant())
        {
            if (c is >= 'a' and <= 'z' or '\'')
            {
                sb.Append(c);
                lastWasSpace = false;
            }
            else if (char.IsWhiteSpace(c) && !lastWasSpace && sb.Length > 0)
            {
                sb.Append(' ');
                lastWasSpace = true;
            }
        }

        // Trim trailing space
        if (sb.Length > 0 && sb[^1] == ' ')
            sb.Length--;

        return sb.ToString();
    }
}
