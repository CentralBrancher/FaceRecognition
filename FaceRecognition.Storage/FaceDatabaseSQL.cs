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

        var sql = @"
            IF EXISTS (SELECT 1 FROM Faces WHERE CroppedPath = @CroppedPath)
            BEGIN
                UPDATE Faces
                SET
                    PersonName   = @PersonName,
                    Embedding    = @Embedding,
                    EmbeddingDim = @EmbeddingDim,
                    ModelName    = @ModelName,
                    Similarity = @Similarity
                WHERE CroppedPath = @CroppedPath
            END
            ELSE
            BEGIN
                INSERT INTO Faces (PersonName, Embedding, EmbeddingDim, ModelName, CroppedPath, Similarity)
                VALUES (@PersonName, @Embedding, @EmbeddingDim, @ModelName, @CroppedPath, @Similarity)
            END";

        await connection.ExecuteAsync(sql, new
        {
            PersonName = face.Label,
            Embedding = FloatArrayToByteArray(face.Embedding),
            EmbeddingDim = face.Embedding.Length,
            ModelName = _modelName,
            CroppedPath = face.CroppedImagePath,
            Similarity = face.MatchSimilarity
        });
    }

    public async Task<IEnumerable<Face>> GetAllFacesAsync(string filePath)
    {
        var faces = new List<Face>();

        using var conn = new SqlConnection(_connectionString);
        await conn.OpenAsync();

        using var cmd = new SqlCommand(
            $"SELECT PersonName, Embedding, CroppedPath FROM Faces", conn);

        using var reader = await cmd.ExecuteReaderAsync();
        while (await reader.ReadAsync())
        {
            if (reader.GetString(2) == filePath) continue;

            faces.Add(new Face
            {
                Label = reader.GetString(0),
                Embedding = ByteArrayToFloatArray((byte[])reader[1])
            });
        }

        return faces;
    }

    private static byte[] FloatArrayToByteArray(float[] floats)
    {
        var bytes = new byte[floats.Length * sizeof(float)];
        Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static float[] ByteArrayToFloatArray(byte[] bytes)
    {
        if (bytes.Length % sizeof(float) != 0)
            throw new InvalidOperationException("Invalid embedding byte length.");

        var floats = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
        return floats;
    }
}
