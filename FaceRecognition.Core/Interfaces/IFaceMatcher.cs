using FaceRecognition.Core.Model;

namespace FaceRecognition.Core.Interfaces;

public interface IFaceMatcher
{
    MatchResult Match(float[] embedding, IEnumerable<Face> knownFaces);
}
