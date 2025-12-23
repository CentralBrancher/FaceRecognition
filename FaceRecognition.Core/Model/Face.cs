using SixLabors.ImageSharp;

namespace FaceRecognition.Core.Model;

public class Face
{
    public string Label { get; set; } = Guid.NewGuid().ToString();
    public Rectangle BoundingBox { get; set; }
    public required float[] Embedding { get; set; }
}
