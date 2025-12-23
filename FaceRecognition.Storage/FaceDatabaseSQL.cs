using Dapper;
using FaceRecognition.Core.Interfaces;
using FaceRecognition.Core.Model;
using Microsoft.Data.SqlClient;

namespace FaceRecognition.Storage;

public class FaceDatabaseSQL(string connectionString, string modelName) : IFaceDatabase
{
    private readonly string _connectionString = connectionString;
    private readonly string _modelName = modelName;

    public async Task AddFaceAsync(Face face)
    {
        if (face.Embedding == null || face.Embedding.Length == 0)
            throw new ArgumentException("Face embedding cannot be null or empty", nameof(face));

        using var connection = new SqlConnection(_connectionString);
        await connection.OpenAsync();

        var sql = @"INSERT INTO Faces (PersonName, Embedding, EmbeddingDim, ModelName)
            VALUES (@PersonName, @Embedding, @EmbeddingDim, @ModelName)";

        await connection.ExecuteAsync(sql, new
        {
            PersonName = face.Label,
            Embedding = FloatArrayToByteArray(face.Embedding),
            EmbeddingDim = face.Embedding.Length,
            ModelName = _modelName
        });
    }

    private static byte[] FloatArrayToByteArray(float[] floats)
    {
        var bytes = new byte[floats.Length * sizeof(float)];
        Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);
        return bytes;
    }
}
