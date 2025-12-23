using FaceRecognition.Core.Interfaces;
using FaceRecognition.Core.Model;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace FaceRecognition.App.Services;

public class FaceRecognitionService(IFaceDetector detector, IFaceEmbedder embedder, IFaceDatabase database)
{
    private readonly IFaceDetector _detector = detector;
    private readonly IFaceEmbedder _embedder = embedder;
    private readonly IFaceDatabase _database = database;

    public async Task<IEnumerable<Face>> ProcessImageAsync(Image<Rgba32> image)
    {
        // 1. Detect faces
        var detectedFaces = _detector.DetectFaces(image);

        var facesWithEmbeddings = new List<Face>();

        foreach (var face in detectedFaces)
        {
            // 2. Crop face
            var faceCrop = CropFace(image, face.BoundingBox);

            // 3. Get embedding
            face.Embedding = _embedder.GetEmbedding(faceCrop);

            // 4. Save to DB
            await _database.AddFaceAsync(face);

            facesWithEmbeddings.Add(face);
        }

        return facesWithEmbeddings;
    }

    private static byte[] CropFace(Image<Rgba32> image, Rectangle bbox)
    {
        using var cropped = image.Clone(ctx => ctx.Crop(bbox));
        using var ms = new MemoryStream();
        cropped.SaveAsJpeg(ms);
        return ms.ToArray();
    }
}