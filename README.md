# VeriPix

**Enhanced Framework for Generalizable Synthetic Image Forgery Detection**

*See beyond the pixels.*

VeriPix is a minor research project at NIT Jalandhar (Department of Information Technology). It studies whether a detector trained on one image generator can still detect images from generators it has never seen.

## Team
- Pranshu
- Darshit
- Maninder Saini
- Mentor: Dr. Jaspal Kaur Saini, Assitant Professor (IT Dept)

## Problem
Generative models such as GANs and diffusion models can produce highly realistic images. Many detectors work well on the generators they were trained on but perform worse on unseen ones. VeriPix investigates this generalization gap.

## Scope
VeriPix targets **AI-generated (synthetic) images**. It does not claim to detect copy-move, splicing or retouching.

## Objectives
- Study the FatFormer architecture
- Prepare real and AI-generated image datasets
- Run the published model as a baseline
- Evaluate with accuracy, precision, recall, F1, AP and ROC-AUC
- Test generalization to unseen GANs and diffusion models
- Build a web interface in a later phase

## Baseline: FatFormer
VeriPix builds on **FatFormer** (Liu et al., CVPR 2024), which adapts a CLIP vision transformer with two components:

- **FAA (Forgery-Aware Adapter):** extracts local image artifacts and frequency clues using wavelet (DWT) decomposition, with attention within and across frequency bands.
- **LGA (Language-Guided Alignment):** uses image features and real/fake text prompts with a contrastive objective to improve the learned representation.

Official repository: https://github.com/Michel-liu/FatFormer

---

## System Design

> The diagrams below show the **planned** design. The web app and our own training results do not exist yet; see [Project status](#project-status).

### Use case diagram

```mermaid
flowchart LR
    U(["User"])
    R(["Researcher / Student"])

    subgraph VeriPix["VeriPix System"]
        UC1["Upload image"]
        UC2["View Real / AI-generated prediction"]
        UC3["View probability and processing time"]
        UC4["Read model limitations"]
        UC5["Prepare dataset and splits"]
        UC6["Train / load FatFormer model"]
        UC7["Evaluate on seen and unseen generators"]
        UC8["View metrics and confusion matrix"]
    end

    U --> UC1
    U --> UC2
    U --> UC3
    U --> UC4
    R --> UC5
    R --> UC6
    R --> UC7
    R --> UC8
    UC1 -.->|includes| UC2
    UC2 -.->|includes| UC3
```

### System architecture

```mermaid
flowchart TB
    subgraph FE["Frontend - React + Tailwind CSS"]
        A1["Upload and preview"]
        A2["Result card"]
        A3["About / limitations page"]
    end

    subgraph BE["Backend - FastAPI"]
        B1["POST /predict"]
        B2["Image validation"]
        B3["Preprocessing"]
        B4["Inference service"]
        B5["GET /health"]
    end

    subgraph ML["Model - FatFormer"]
        C1["CLIP ViT image encoder"]
        C2["Forgery-Aware Adapter (FAA)"]
        C3["Language-Guided Alignment (LGA)"]
        C4["Real / Fake classifier"]
    end

    subgraph DB["Optional storage (later phase)"]
        D1[("Prediction history")]
    end

    A1 -->|"image file (HTTP)"| B1
    B1 --> B2 --> B3 --> B4
    B4 --> C1
    C1 --> C2 --> C3 --> C4
    C4 -->|"label + probability"| B4
    B4 -->|"JSON response"| A2
    B4 -.-> D1
```

### FatFormer model pipeline

```mermaid
flowchart LR
    IMG["Input image"] --> PRE["Resize 256, crop 224, normalize"]
    PRE --> PE["Patch embedding"]
    PE --> S1["ViT stage 1"]
    S1 --> F1["FAA"]
    F1 --> S2["ViT stage 2"]
    S2 --> F2["FAA"]
    F2 --> SN["ViT stage N"]
    SN --> FEAT["Image CLS token + patch tokens"]

    subgraph TXT["Text side"]
        P["Learnable context embeddings + class word real/fake"]
        TE["CLIP text encoder"]
        P --> TE
    end

    FEAT --> PBE["Patch-based enhancer"]
    PBE --> P
    TE --> TGI["Text-guided interactor"]
    FEAT --> TGI
    FEAT --> SIM["Cosine similarity with real/fake text embeddings"]
    TGI --> SIM
    SIM --> SOFT["Softmax"]
    SOFT --> OUT["Real or AI-generated"]
```

### Inside one Forgery-Aware Adapter (FAA)

```mermaid
flowchart LR
    IN["ViT stage features"] --> IMGB
    IN --> DWT

    subgraph IMGB["Image branch"]
        I1["Conv"] --> I2["ReLU"] --> I3["Conv"]
    end

    subgraph FREQ["Frequency branch"]
        DWT["DWT: LL, LH, HL, HH"] --> ATT["Intra-band and inter-band attention"]
        ATT --> FFN["FFN"] --> IDWT["Inverse DWT"]
    end

    I3 --> SUM["image + lambda x frequency"]
    IDWT --> SUM
    SUM --> OUTF["Adapted features to next ViT stage"]
```

###  Sequence Diagram: Prediction flow (planned web app)

```mermaid
sequenceDiagram
    actor User
    participant FE as React Frontend
    participant API as FastAPI Backend
    participant PRE as Preprocessor
    participant M as FatFormer Model

    User->>FE: Select / drop image
    FE->>FE: Show preview
    FE->>API: POST /predict (image file)
    API->>API: Validate type and size
    alt Invalid file
        API-->>FE: 400 error message
        FE-->>User: Show error
    else Valid file
        API->>PRE: Resize, crop, normalize
        PRE-->>API: Tensor
        API->>M: Forward pass
        M-->>API: P(fake)
        API->>API: Apply threshold, measure time
        API-->>FE: JSON (label, probability, model, time)
        FE-->>User: Result card + disclaimer
    end
```

### Class diagram (planned code structure)

```mermaid
classDiagram
    class ImageValidator {
        +allowed_types: list
        +max_size_mb: int
        +validate(file) bool
    }

    class Preprocessor {
        +resize: int
        +crop: int
        +mean: list
        +std: list
        +transform(image) Tensor
    }

    class FatFormerModel {
        +clip_encoder: CLIPImageEncoder
        +faa: ForgeryAwareAdapter
        +lga: LanguageGuidedAlignment
        +forward(x) Tensor
    }

    class ForgeryAwareAdapter {
        +image_branch
        +frequency_branch
        +lambda_scale
        +forward(features) Tensor
    }

    class LanguageGuidedAlignment {
        +context_embeddings
        +patch_based_enhancer
        +text_guided_interactor
        +similarity(img, text) Tensor
    }

    class Predictor {
        +model: FatFormerModel
        +threshold: float
        +predict(image) PredictionResult
    }

    class PredictionResult {
        +label: str
        +fake_probability: float
        +model_name: str
        +processing_time_ms: float
    }

    class Evaluator {
        +compute_metrics(y_true, y_score) dict
        +confusion_matrix(y_true, y_pred)
        +evaluate_generator(name, data)
    }

    FatFormerModel *-- ForgeryAwareAdapter
    FatFormerModel *-- LanguageGuidedAlignment
    Predictor --> FatFormerModel
    Predictor --> Preprocessor
    Predictor ..> PredictionResult : returns
    Predictor --> ImageValidator : uses before inference
    Evaluator --> Predictor : scores test sets
```

### ER diagram (prediction history, later phase)

```mermaid
erDiagram
    PREDICTION {
        int id PK
        string filename
        string label
        float fake_probability
        string model_name
        float processing_time_ms
        datetime created_at
    }
```

All diagrams are also collected in [`docs/DIAGRAMS.md`](docs/DIAGRAMS.md).

---

## Research Methodology

### Activity diagram

```mermaid
flowchart TD
    S(["Start"]) --> L["Literature review and scope"]
    L --> D["Select datasets: ProGAN train, unseen GANs and diffusion models"]
    D --> SP{"Split by generator?"}
    SP -->|"No"| FIX["Fix split to avoid leakage"]
    FIX --> SP
    SP -->|"Yes"| ENV["Set up FatFormer environment"]
    ENV --> OK{"Runs on available GPU?"}
    OK -->|"No"| DOC["Document blocker, use released checkpoint / smaller subset"]
    OK -->|"Yes"| TR["Train or load FatFormer"]
    DOC --> TR
    TR --> VAL["Validate and select checkpoint (validation data only)"]
    VAL --> TEST["Test: seen generator"]
    TEST --> UNS["Test: unseen GANs and diffusion models"]
    UNS --> MET["Compute ACC, precision, recall, F1, AP, ROC-AUC, confusion matrix"]
    MET --> AN["Analyse generalization gaps and failure cases"]
    AN --> APP["Build FastAPI + React demo"]
    APP --> E(["Report and presentation"])
```

### Data flow: train / test separation

```mermaid
flowchart LR
    subgraph TRAIN["Training generator"]
        PG["ProGAN real + fake"]
    end

    PG --> TS["Train set"]
    PG --> VS["Validation set"]
    PG --> HT["Held-out seen test"]

    subgraph UNSEEN["Never used in training"]
        G1["Other GANs: StyleGAN, BigGAN, CycleGAN ..."]
        G2["Diffusion: LDM, Glide, DALL-E ..."]
    end

    TS --> MOD["FatFormer training"]
    VS --> MOD
    MOD --> CK["Selected checkpoint"]
    CK --> E1["Evaluate: seen"]
    CK --> E2["Evaluate: unseen GANs"]
    CK --> E3["Evaluate: unseen diffusion"]
    HT --> E1
    G1 --> E2
    G2 --> E3
    E1 --> RES["Results table per generator"]
    E2 --> RES
    E3 --> RES
```

## Evaluation
- Metrics: accuracy, precision, recall, F1, AP, ROC-AUC, confusion matrix, inference time
- Label convention: real = 0, fake = 1
- Generators are separated between training and testing, so unseen-generator results are meaningful

## Project status
| Item | Status |
|---|---|
| Literature study | In progress |
| FatFormer environment setup | Planned |
| Baseline inference | Planned |
| Cross-generator evaluation | Planned |
| Training | Planned |
| Web app (FastAPI + React) | Planned |

> Numbers reported in the FatFormer paper belong to its authors, not to VeriPix. Results will be added here only after we reproduce them. "Enhanced" will be claimed only for changes we implement and evaluate.

## Quick start (Kaggle)
```bash
git clone https://github.com/<your-username>/VeriPix.git
cd VeriPix
git clone https://github.com/Michel-liu/FatFormer.git
```
Then follow the FatFormer README for dependencies, the checkpoint and the evaluation command.

## Planned tech stack
Python, PyTorch, CLIP, pytorch_wavelets, scikit-learn, FastAPI, React, Tailwind CSS

## References
- Liu et al., *Forgery-aware Adaptive Transformer for Generalizable Synthetic Image Detection*, CVPR 2024.
- Ojha et al., *Towards Universal Fake Image Detectors that Generalize Across Generative Models*, CVPR 2023.
- Wang et al., *CNN-generated images are surprisingly easy to spot... for now*, CVPR 2020.

