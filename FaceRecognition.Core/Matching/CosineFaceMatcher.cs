using FaceRecognition.Core.Interfaces;
using FaceRecognition.Core.Model;

namespace FaceRecognition.Core.Matching;

public sealed class CosineFaceMatcher(float threshold = 0.5f) : IFaceMatcher
{
    private readonly float _threshold = threshold;

    public MatchResult Match(float[] embedding, IEnumerable<Face> knownFaces)
    {
        string? bestLabel = null;
        float bestScore = float.MinValue;

        foreach (var face in knownFaces)
        {
            if (face.Embedding is null)
                continue;

            var score = Dot(embedding, face.Embedding);
            if (score > bestScore)
            {
                bestScore = score;
                bestLabel = face.Label;
            }
        }

        return new MatchResult
        {
            IsMatch = bestScore >= _threshold,
            MatchedLabel = bestScore >= _threshold ? bestLabel : null,
            Similarity = bestScore
        };
    }

    private static float Dot(float[] a, float[] b)
    {
        float sum = 0f;
        for (int i = 0; i < a.Length; i++)
            sum += a[i] * b[i];
        return sum;
    }
}
