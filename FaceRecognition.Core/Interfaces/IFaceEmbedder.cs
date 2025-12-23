namespace FaceRecognition.Core.Interfaces;

public interface IFaceEmbedder
{
    float[] GetEmbedding(byte[] faceImageData);
}
