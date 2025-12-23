# AllMpnetBaseV2Sharp

C# implementation of Sentence Transformers **all-mpnet-base-v2**.

Built as a modern **.NET 8** class library using **ONNX Runtime** and
**HuggingFace tokenizers** for correctness and parity with the original Python model.

---

## 📦 NuGet

[AllMpnetBaseV2Sharp](https://www.nuget.org/packages/AllMpnetBaseV2Sharp)

❌ **The NuGet package does not include the ONNX model or tokenizer.json.**
You will need to download them manually from [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) on Hugging Face.

By default, the embedder looks for these files in the ./model folder:
```./model/model.onnx```
```./model/tokenizer.json```

You can also specify custom file locations by setting the paths in MpnetOptions.

---

## 🚀 How to Use

### Single sentence

```csharp
using AllMpnetBaseV2Sharp;

var sentence = "This is an example sentence";

using var embedder = new AllMpnetBaseV2Embedder();
var embedding = embedder.Encode(sentence);
```
### Multiple sentences

```csharp
using AllMpnetBaseV2Sharp;

string[] sentences =
{
    "This is an example sentence",
    "Here is another"
};

using var embedder = new AllMpnetBaseV2Embedder();
var embeddings = embedder.Encode(sentences);
```
### Custom ONNX model (advanced)

```csharp
var options = new MpnetOptions
{
    ModelPath = "path/to/model.onnx",
    TokenizerPath = "path/to/tokenizer.json"
};

using var embedder = new AllMpnetBaseV2Embedder(options);
var embedding = embedder.Encode("This is an example sentence");
```