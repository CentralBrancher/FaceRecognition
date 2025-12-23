using FaceRecognition.Core.Interfaces;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace FaceRecognition.ONNX;

public class FaceEmbedderONNX : IFaceEmbedder
{
    private readonly InferenceSession _session;
    private const int INPUT_WIDTH = 112;
    private const int INPUT_HEIGHT = 112;

    public FaceEmbedderONNX(string modelPath)
    {
        _session = new InferenceSession(modelPath);
    }

    public FaceEmbedderONNX(string modelPath, SessionOptions options)
    {
        _session = new InferenceSession(modelPath, options);
    }

    public float[] GetEmbedding(byte[] faceImageData)
    {
        using var image = Image.Load<Rgba32>(faceImageData);
        var tensor = Preprocess(image);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_1", tensor)
        };

        using var results = _session.Run(inputs);
        var output = results[0].AsEnumerable<float>().ToArray();

        // L2 normalize embedding
        var norm = Math.Sqrt(output.Sum(x => x * x));
        return [.. output.Select(x => (float)(x / norm))];
    }

    private static DenseTensor<float> Preprocess(Image<Rgba32> image)
    {
        var resized = image.Clone(ctx => ctx.Resize(INPUT_WIDTH, INPUT_HEIGHT));
        var tensor = new DenseTensor<float>([1, INPUT_HEIGHT, INPUT_WIDTH, 3]); // NHWC

        for (int y = 0; y < INPUT_HEIGHT; y++)
        {
            for (int x = 0; x < INPUT_WIDTH; x++)
            {
                var pixel = resized[x, y];
                tensor[0, y, x, 0] = pixel.B / 255f;
                tensor[0, y, x, 1] = pixel.G / 255f;
                tensor[0, y, x, 2] = pixel.R / 255f;
            }
        }

        return tensor;
    }
}
