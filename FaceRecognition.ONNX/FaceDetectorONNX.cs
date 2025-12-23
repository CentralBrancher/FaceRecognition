using FaceRecognition.Core.Model;
using FaceRecognition.Core.Interfaces;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.PixelFormats;

namespace FaceRecognition.ONNX;

public class FaceDetectorONNX : IFaceDetector
{
    private readonly InferenceSession _session;
    public FaceDetectorONNX(string modelPath)
    {
        _session = new InferenceSession(modelPath);
    }

    public FaceDetectorONNX(string modelPath, SessionOptions options)
    {
        _session = new InferenceSession(modelPath, options);
    }

    private const int INPUT_WIDTH = 320;
    private const int INPUT_HEIGHT = 240;
    private const float SCORE_THRESHOLD = 0.7f;

    public IEnumerable<Face> DetectFaces(Image<Rgba32> image)
    {
        var inputTensor = Preprocess(image);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", inputTensor)
        };

        using var results = _session.Run(inputs);

        var boxes = results.First(r => r.Name.Contains("boxes")).AsTensor<float>();
        var scores = results.First(r => r.Name.Contains("scores")).AsTensor<float>();

        return DecodeResults(boxes, scores, image.Width, image.Height);
    }

    // -------------------- PREPROCESS --------------------

    private static DenseTensor<float> Preprocess(Image<Rgba32> image)
    {
        var resized = image.Clone(ctx =>
            ctx.Resize(INPUT_WIDTH, INPUT_HEIGHT));

        var tensor = new DenseTensor<float>([1, 3, INPUT_HEIGHT, INPUT_WIDTH]);

        for (int y = 0; y < INPUT_HEIGHT; y++)
        {
            for (int x = 0; x < INPUT_WIDTH; x++)
            {
                var pixel = resized[x, y];

                // BGR + normalization
                tensor[0, 0, y, x] = (pixel.B - 127f) / 128f;
                tensor[0, 1, y, x] = (pixel.G - 127f) / 128f;
                tensor[0, 2, y, x] = (pixel.R - 127f) / 128f;
            }
        }

        return tensor;
    }

    // -------------------- POSTPROCESS --------------------

    private static List<Face> DecodeResults(
        Tensor<float> boxes,
        Tensor<float> scores,
        int imageWidth,
        int imageHeight)
    {
        var faces = new List<Face>();

        for (int i = 0; i < scores.Dimensions[1]; i++)
        {
            float confidence = scores[0, i, 1];
            if (confidence < SCORE_THRESHOLD)
                continue;

            float xmin = boxes[0, i, 0] * imageWidth;
            float ymin = boxes[0, i, 1] * imageHeight;
            float xmax = boxes[0, i, 2] * imageWidth;
            float ymax = boxes[0, i, 3] * imageHeight;

            var rect = Rectangle.FromLTRB(
                (int)xmin,
                (int)ymin,
                (int)xmax,
                (int)ymax);

            faces.Add(new Face
            {
                BoundingBox = rect,
                Embedding = []
            });
        }

        return faces;
    }
}
