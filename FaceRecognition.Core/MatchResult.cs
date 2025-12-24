namespace FaceRecognition.Core;

public sealed class MatchResult
{
    public bool IsMatch { get; init; }
    public string? MatchedLabel { get; init; }
    public float Similarity { get; init; }
}
