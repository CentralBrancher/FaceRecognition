# FaceRecognition (.NET / C#)

A **C#/.NET–based face recognition system** built from first principles — no Python runtime, minimal dependencies, explicit architecture.

This project implements an **end-to-end face recognition pipeline** using:

- **UltraFace (ONNX)** for face detection  
- **ArcFace (ONNX)** for face embeddings  
- **Cosine similarity** for matching  
- **SQL Server** for persistence  

The goal is **clarity, control, and correctness**, not a black-box demo.

---

## Features

- Face detection from images (UltraFace ONNX)
- Face embedding extraction (ArcFace ONNX, 512-D)
- Deterministic cosine similarity matching
- SQL Server–backed embedding storage
- Clean separation of concerns
- Fully testable (.NET + xUnit)

---

## High-Level Pipeline

```bash
Image
└─► UltraFace (detect)
└─► Crop face
└─► ArcFace (embed, 512D)
└─► SQL storage
└─► Cosine similarity match
```

---

## Repository Structure

```bash
FaceRecognition/
│
├── FaceRecognition.sln
│
├──models/
│ ├── version-RFB-320.onnx # UltraFace
│ └── arc.onnx # ArcFace
│
├── FaceRecognition.App/
│ ├── Program.cs
│ ├── Services/
│ │ └── FaceRecognitionService.cs
│ ├── faces/ # Output faces
│ └── images/ # Input images
│
├── FaceRecognition.Core/ # Domain + contracts
│ ├── Interfaces/
│ │ ├── IFaceDetector.cs
│ │ ├── IFaceEmbedder.cs
│ │ ├── IFaceEmbedder.cs
│ │ └── IFaceMatcher.cs
│ ├── Model/
│ │ └── Face.cs
│ ├── Matching/
│ │ └── CosineFaceMatcher.cs
│ ├── MatchResult.cs
│ └── Utils/
│ └── ImageExtensions.cs
│
├── FaceRecognition.ONNX/ # ONNX Runtime bindings
│ ├── FaceDetectorONNX.cs
│ └── FaceEmbedderONNX.cs
│
├── FaceRecognition.Storage/ # SQL Server persistence
│ └── FaceDatabaseSQL.cs
│
├── FaceRecognition.Tests/ # xUnit tests
│ ├── FaceRecognitionTests.cs
│ └── images/
│
└── README.md
```

---

## Requirements

- **.NET 9 SDK**
- **SQL Server / SQL Express**
- Windows x64 (ONNX Runtime CPU)

---

## Dependencies

Minimal and explicit:

- `Microsoft.ML.OnnxRuntime`
- `SixLabors.ImageSharp`
- `Microsoft.Data.SqlClient`
- `xUnit` (tests)

---

## Models Used

### UltraFace (Face Detection)

- Lightweight, edge-friendly
- ONNX opset 9
- Input: RGB image
- Output: bounding boxes + confidence

### ArcFace (Face Embeddings)

- 512-dimensional embeddings
- L2-normalized
- Cosine similarity compatible
- Input: aligned 112×112 face crop

---

## Database Schema

Example SQL table:

```sql
CREATE TABLE Faces (
    Id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    PersonName     NVARCHAR(128) NOT NULL,
    CroppedPath    NVARCHAR(512) NOT NULL UNIQUE,
    Embedding      VARBINARY(MAX) NOT NULL,
    EmbeddingDim   INT NOT NULL,
    ModelName      NVARCHAR(64) NOT NULL,
    Similarity     FLOAT NULL,
    CreatedAt      DATETIME2 DEFAULT SYSDATETIME()
);
```

---

## Matching Logic

- Embeddings are L2-normalized
- Similarity = dot product
- Threshold is configurable

Recommended starting threshold: Cosine similarity ≈ 0.60

Similarity	Meaning:
- ≥ 0.80	Same person (high confidence)
- 0.60 – 0.80	Likely same person
- < 0.60	Different person

There is no universal ArcFace threshold — tune for your data.

---

## Running the App

Run:

```bash
dotnet run --project FaceRecognition.App
```

The app will:
- Detect faces
- Crop & save face images
- Generate embeddings
- Store them in SQL
- Attempt recognition against existing entries
