using FaceRecognition.ONNX;
using FaceRecognition.Storage;
using FaceRecognition.App.Services;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Microsoft.ML.OnnxRuntime;
using FaceRecognition.Core.Matching;

namespace FaceRecognition.Tests;

public class FaceDetectionTests
{
    [Fact]
    public async Task Detects_And_Stores_Faces_From_Test_Images()
    {
        // Arrange
        string basePath = AppContext.BaseDirectory;
        string imagesPath = Path.Combine(basePath, "images");

        string solutionRoot = Path.Combine(basePath, "../../../..");
        string modelsPath = Path.Combine(solutionRoot, "models");

        var options = new SessionOptions
        {
            LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR  // ORT_LOGGING_LEVEL_ERROR = 3
        };

        var detector = new FaceDetectorONNX(
            Path.Combine(modelsPath, "version-RFB-320.onnx"), options);

        var embedder = new FaceEmbedderONNX(
            Path.Combine(modelsPath, "arc.onnx"), options);

        var database = new FaceDatabaseSQL(
            "Server=.\\SQLEXPRESS;Database=FaceRecognition;Trusted_Connection=True;TrustServerCertificate=True", 
            "arcface_512");

        var matcher = new CosineFaceMatcher(threshold: (float)0.65);

        var recognitionService = new FaceRecognitionService(
            detector, embedder, database, matcher);

        int totalFaces = 0;

        // Act
        foreach (var imageFile in Directory.GetFiles(imagesPath, "*.jpg"))
        {
            using var image = Image.Load<Rgba32>(imageFile);
            string imageName = Path.GetFileNameWithoutExtension(imageFile);
            var results = await recognitionService.ProcessImageAsync(image, imageName);
            totalFaces += results.Count();
        }

        // Assert
        Assert.True(totalFaces > 0, "No faces were detected.");
    }
}
