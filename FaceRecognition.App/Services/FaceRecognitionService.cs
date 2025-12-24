using FaceRecognition.Core.Interfaces;
using FaceRecognition.Core.Model;
using FaceRecognition.Core.Utils;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace FaceRecognition.App.Services;

public class FaceRecognitionService(IFaceDetector detector, IFaceEmbedder embedder, IFaceDatabase database, IFaceMatcher matcher)
{
    private readonly IFaceDetector _detector = detector;
    private readonly IFaceEmbedder _embedder = embedder;
    private readonly IFaceDatabase _database = database;
    private readonly IFaceMatcher _matcher = matcher;    

    public async Task<IEnumerable<Face>> ProcessImageAsync(Image<Rgba32> image, string imageName)
    {
        var detectedFaces = _detector.DetectFaces(image);
        var facesWithEmbeddings = new List<Face>();
        string facesFolder = Path.Combine(AppContext.BaseDirectory, "faces");

        if (!Directory.Exists(facesFolder))
            Directory.CreateDirectory(facesFolder);
    
        int faceIndex = 0;

        foreach (var face in detectedFaces)
        {
            // 1. Crop face
            var faceCrop = CropFace(image, face.BoundingBox);

            // 2. Save cropped face
            string faceFile = Path.Combine(facesFolder, $"{imageName}_face_{faceIndex}.jpg");
            await File.WriteAllBytesAsync(faceFile, faceCrop);
            face.CroppedImagePath = faceFile;

            // 3. Get embedding
            face.Embedding = _embedder.GetEmbedding(faceCrop);

            // 4. Check matches
            var knownFaces = await _database.GetAllFacesAsync(faceFile);
            var match = _matcher.Match(face.Embedding, knownFaces);

            if (match.IsMatch)
            {
                face.Label = match.MatchedLabel!;
                face.MatchSimilarity = match.Similarity;
            }

            // 5. Save to DB
            await _database.AddFaceAsync(face);

            facesWithEmbeddings.Add(face);

            faceIndex++;
        }

        return facesWithEmbeddings;
    }

    private static byte[] CropFace(Image<Rgba32> image, Rectangle bbox)
    {
        using var cropped = image.Clone(ctx => ctx.Crop(bbox));
        return cropped.ToJpegBytes();
    }
}