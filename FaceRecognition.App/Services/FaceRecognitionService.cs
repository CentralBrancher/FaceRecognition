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
        var detectedFaces = _detector.DetectFaces(image);
        var facesWithEmbeddings = new List<Face>();
        string facesFolder = Path.Combine(AppContext.BaseDirectory, "faces");

        if (!Directory.Exists(facesFolder))
            Directory.CreateDirectory(facesFolder);

        foreach (var face in detectedFaces)
        {
            // 1. Crop face
            var faceCrop = CropFace(image, face.BoundingBox);

            // 2. Save cropped face
            string faceFile = Path.Combine(facesFolder, $"face_{Guid.NewGuid()}.jpg");
            await File.WriteAllBytesAsync(faceFile, faceCrop);
            face.CroppedImagePath = faceFile;

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