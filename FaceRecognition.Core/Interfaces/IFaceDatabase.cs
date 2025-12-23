using FaceRecognition.Core.Model;

namespace FaceRecognition.Core.Interfaces;

public interface IFaceDatabase
{
    Task AddFaceAsync(Face face);
}
