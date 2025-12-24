using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace FaceRecognition.Core.Utils;

public static class ImageExtensions
{
    public static byte[] ToJpegBytes(this Image<Rgba32> image)
    {
        using var ms = new MemoryStream();
        image.SaveAsJpeg(ms);
        return ms.ToArray();
    }
}
