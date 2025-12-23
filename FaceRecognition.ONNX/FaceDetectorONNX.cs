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
    private const int INPUT_WIDTH = 320;
    private const int INPUT_HEIGHT = 240;
    private const float SCORE_THRESHOLD = 0.7f;
    private const float IOU_THRESHOLD = 0.5f;

    public FaceDetectorONNX(string modelPath)
    {
        _session = new InferenceSession(modelPath);
    }

    public FaceDetectorONNX(string modelPath, SessionOptions options)
    {
        _session = new InferenceSession(modelPath, options);
    }

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
        var resized = image.Clone(ctx => ctx.Resize(INPUT_WIDTH, INPUT_HEIGHT));
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
        // Step 1: Collect boxes that pass confidence threshold
        var rawBoxes = new List<(Rectangle rect, float score)>();

        for (int i = 0; i < scores.Dimensions[1]; i++)
        {
            float confidence = scores[0, i, 1]; // face score
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
                (int)ymax
            );

            rawBoxes.Add((rect, confidence));
        }

        // Step 2: Apply Non-Max Suppression to remove overlapping boxes
        var nmsBoxes = NonMaxSuppression(rawBoxes, IOU_THRESHOLD);

        // Step 3: Convert to Face objects
        var faces = nmsBoxes.Select(rect => new Face
        {
            BoundingBox = rect,
            Embedding = null // ArcFace embedding will be assigned later
        }).ToList();

        return faces;
    }

    // -------------------- NON-MAX SUPPRESSION --------------------
    private static List<Rectangle> NonMaxSuppression(List<(Rectangle rect, float score)> boxes, float iouThreshold)
    {
        var selected = new List<Rectangle>();

        // Sort boxes by descending score
        var sorted = boxes.OrderByDescending(b => b.score).ToList();

        while (sorted.Count != 0)
        {
            var (rect, score) = sorted[0];
            selected.Add(rect);
            sorted.RemoveAt(0);

            sorted = [.. sorted.Where(b => IoU(rect, b.rect) < iouThreshold)];
        }

        return selected;
    }

    private static float IoU(Rectangle a, Rectangle b)
    {
        int x1 = Math.Max(a.Left, b.Left);
        int y1 = Math.Max(a.Top, b.Top);
        int x2 = Math.Min(a.Right, b.Right);
        int y2 = Math.Min(a.Bottom, b.Bottom);

        int interArea = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
        int unionArea = a.Width * a.Height + b.Width * b.Height - interArea;

        return unionArea == 0 ? 0 : (float)interArea / unionArea;
    }
}
