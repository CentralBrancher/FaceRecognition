using FaceRecognition.ONNX;
using FaceRecognition.Storage;
using FaceRecognition.App.Services;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace FaceRecognition.App;

class Program
{
    static async Task Main()
    {
        // Suppress ONNX Runtime warnings
        Environment.SetEnvironmentVariable("ORT_LOG_SEVERITY_LEVEL", "3");

        Console.WriteLine("Face Recognition Step 1 - Load & Encode Faces");

        // Paths
        string imagesPath = Path.Combine(AppContext.BaseDirectory, "images");
        string modelsPath = Path.Combine(AppContext.BaseDirectory, "models");

        // Initialize services
        var detector = new FaceDetectorONNX(Path.Combine(modelsPath, "version-RFB-320.onnx"));
        var embedder = new FaceEmbedderONNX(Path.Combine(modelsPath, "arc.onnx"));
        var database = new FaceDatabaseSQL("Server=.\\SQLEXPRESS;Database=FaceRecognition;Trusted_Connection=True;TrustServerCertificate=True", 
            "arcface_512");
        var recognitionService = new FaceRecognitionService(detector, embedder, database);

        // Process each image
        foreach (var imageFile in Directory.GetFiles(imagesPath, "*.jpg"))
        {
            Console.WriteLine($"Processing: {Path.GetFileName(imageFile)}");

            using var image = Image.Load<Rgba32>(imageFile);
            var results = await recognitionService.ProcessImageAsync(image);

            foreach (var face in results)
            {
                Console.WriteLine($"Saved face for label: {face.Label}");
            }
        }

        Console.WriteLine("Processing complete.");
    }
}
