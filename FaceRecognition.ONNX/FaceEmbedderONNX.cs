using FaceRecognition.Core.Interfaces;

namespace FaceRecognition.ONNX;

public class FaceEmbedderONNX : IFaceEmbedder
{
    public FaceEmbedderONNX(string modelPath)
    {
        // Load ONNX embedding model
    }

    public float[] GetEmbedding(byte[] faceImageData)
    {
        // Placeholder: return dummy 128-dim vector
        return new float[128];
    }
}
