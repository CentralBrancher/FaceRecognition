using FaceRecognition.ONNX;
using FaceRecognition.Storage;
using FaceRecognition.App.Services;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Microsoft.ML.OnnxRuntime;

namespace FaceRecognition.App;

class Program
{
    static async Task Main()
    {
        Console.WriteLine("Face Recognition Step 1 - Load & Encode Faces");

        // Paths
        string basePath = AppContext.BaseDirectory;
        string imagesPath = Path.Combine(basePath, "images");
        string modelsPath = Path.Combine(basePath, "models");

        // Initialize services
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

        var recognitionService = new FaceRecognitionService(
            detector, embedder, database);

        // Process each image
        foreach (var imageFile in Directory.GetFiles(imagesPath, "*.jpg"))
        {
            Console.WriteLine($"Processing: {Path.GetFileName(imageFile)}");

            using var image = Image.Load<Rgba32>(imageFile);
            string imageName = Path.GetFileNameWithoutExtension(imageFile);
            var results = await recognitionService.ProcessImageAsync(image, imageName);

            foreach (var face in results)
            {
                Console.WriteLine($"Saved face for label: {face.Label}");
            }
        }

        Console.WriteLine("Processing complete.");
    }
}
