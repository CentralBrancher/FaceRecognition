using FaceRecognition.Core.Model;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace FaceRecognition.Core.Interfaces;

public interface IFaceDetector
{
    IEnumerable<Face> DetectFaces(Image<Rgba32> image);
}
