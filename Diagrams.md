# VeriPix Diagrams

All diagrams for **VeriPix: Enhanced Framework for Generalizable Synthetic Image Forgery Detection**, written in Mermaid (GitHub renders them automatically). Save as `docs/DIAGRAMS.md`.

> These show the **planned** design. The web app, backend and our own trained model do not exist yet.

## Contents

| # | Diagram | Type |
|---|---|---|
| 1 | Use case | UML use case |
| 2 | System architecture | Architecture |
| 3, 3a | FatFormer pipeline, FAA internals | Model flow |
| 4 | Prediction flow | UML sequence |
| 5 | Code structure | UML class |
| 6 | Research methodology | UML activity |
| 7 | Train/test data flow | Data flow |
| 8 | Prediction history | ER |
| 9 | Prediction request lifecycle | UML state |
| 10 | Model lifecycle | UML state |
| 11 | Prediction scenario | UML communication (collaboration) |
| 12 | Layered pattern | Architectural pattern |
| 13 | Multi-tier pattern | Architectural pattern |
| 14 | Client-Server pattern | Architectural pattern |
| 15 | MVC | Architectural pattern |
| 16 | Pipes and Filters | Architectural pattern |
| 17 | Singleton | Design pattern |
| 18 | Facade | Design pattern |
| 19 | Factory | Design pattern |
| 20 | Repository | Architectural pattern |

---

## 1. Use Case Diagram

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

---

## 2. System Architecture (Layered / Component View)

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

---

## 3. FatFormer Model Pipeline

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

### 3a. Inside one Forgery-Aware Adapter (FAA)

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

---

## 4. Sequence Diagram: Predicting an Image (web app)

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

---

## 5. Class Diagram (planned code structure)

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

---

## 6. Activity Diagram: Research Methodology

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

---

## 7. Experiment Data Flow (train / test separation)

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

---

## 8. Optional ER Diagram (prediction history, later phase)

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

---

## 9. State Diagram: Prediction Request Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> ImageSelected : user selects image
    ImageSelected --> Validating : user clicks Analyse
    Validating --> Rejected : invalid type or size
    Validating --> Processing : valid image
    Processing --> ResultReady : model returns probability
    Processing --> Failed : server or model error
    Rejected --> Idle : user picks another image
    Failed --> Idle : user retries
    ResultReady --> Idle : user uploads new image
    ResultReady --> [*] : user leaves page
```

---

## 10. State Diagram: Model Lifecycle (backend)

```mermaid
stateDiagram-v2
    [*] --> NotLoaded
    NotLoaded --> Loading : server starts
    Loading --> Ready : checkpoint loaded
    Loading --> LoadFailed : missing or corrupt checkpoint
    LoadFailed --> Loading : fix and restart
    Ready --> Inferring : request received
    Inferring --> Ready : result returned
    Ready --> [*] : server stops
```

---

## 11. Communication (Collaboration) Diagram: Prediction Scenario

Objects are linked and each message is numbered to show the order. This is the same scenario as the sequence diagram (4), drawn as a communication diagram.

```mermaid
flowchart LR
    U(["User"])
    F["Frontend : React UI"]
    A["Backend : FastAPI"]
    V["Validator : ImageValidator"]
    P["Preprocessor"]
    M["Model : FatFormerModel"]

    U -->|"1: selectImage()"| F
    F -->|"2: POST /predict(image)"| A
    A -->|"3: validate(file)"| V
    A -->|"4: transform(image)"| P
    A -->|"5: forward(tensor)"| M
    M -->|"6: P(fake)"| A
    A -->|"7: JSON result"| F
    F -->|"8: showResult()"| U
```

---

## 12. Layered Pattern

Each layer uses only the layer directly below it.

```mermaid
flowchart TB
    L1["Layer 1 - Presentation: React + Tailwind UI"]
    L2["Layer 2 - API: FastAPI routes, validation"]
    L3["Layer 3 - Service: Predictor, Preprocessor"]
    L4["Layer 4 - Model: CLIP + FAA + LGA (FatFormer)"]
    L5[("Layer 5 - Data: checkpoints, prediction history")]

    L1 --> L2 --> L3 --> L4 --> L5
```

---

## 13. Multi-Tier Pattern

```mermaid
flowchart LR
    subgraph T1["Tier 1 - Client (browser)"]
        C1["React app"]
    end
    subgraph T2["Tier 2 - Application server"]
        C2["FastAPI service"]
        C3["Preprocessing and inference logic"]
    end
    subgraph T3["Tier 3 - Model and data"]
        C4["FatFormer checkpoint (GPU)"]
        C5[("History DB, optional")]
    end

    C1 <-->|"HTTP / JSON"| C2
    C2 --> C3
    C3 --> C4
    C2 -.-> C5
```

---

## 14. Client-Server Pattern

```mermaid
flowchart LR
    C1["Client: web browser"]
    C2["Client: API tester (curl / Swagger)"]
    S["Server: FastAPI + FatFormer"]

    C1 -->|"request: POST /predict"| S
    S -->|"reply: label, probability, time"| C1
    C2 -->|"request"| S
    S -->|"reply"| C2
```

---

## 15. MVC Pattern

```mermaid
flowchart LR
    V["View: React components (upload, result card)"]
    C["Controller: FastAPI routes (/predict, /health)"]
    M["Model: FatFormer + Predictor + history data"]

    V -->|"user action: upload image"| C
    C -->|"update / run inference"| M
    M -->|"prediction result"| C
    C -->|"JSON response"| V
```

---

## 16. Pipes and Filters Pattern

Each filter does one job. Pipes carry the data between filters.

```mermaid
flowchart LR
    SRC(["Source: uploaded image"])
    F1["Filter 1: validate type and size"]
    F2["Filter 2: resize to 256"]
    F3["Filter 3: center crop to 224"]
    F4["Filter 4: normalize"]
    F5["Filter 5: FatFormer inference"]
    F6["Filter 6: softmax and threshold"]
    SNK(["Sink: result card"])

    SRC -->|pipe| F1 -->|pipe| F2 -->|pipe| F3 -->|pipe| F4 -->|pipe| F5 -->|pipe| F6 -->|pipe| SNK
```

---

## 17. Singleton Pattern (model loaded once)

Loading a ViT-L/14 model for every request would be too slow, so one shared instance is created at startup.

```mermaid
classDiagram
    class ModelService {
        -instance: ModelService
        -model: FatFormerModel
        -ModelService()
        +get_instance() ModelService
        +predict(tensor) float
    }
    class PredictEndpoint {
        +handle(request)
    }
    PredictEndpoint ..> ModelService : get_instance()
```

---

## 18. Facade Pattern

`Predictor` gives the API one simple method and hides the subsystems behind it.

```mermaid
classDiagram
    class Predictor {
        +predict(file) PredictionResult
    }
    class ImageValidator {
        +validate(file) bool
    }
    class Preprocessor {
        +transform(image) Tensor
    }
    class ModelService {
        +predict(tensor) float
    }
    class ResultFormatter {
        +build(prob, time) PredictionResult
    }
    class ApiRoute {
        +post_predict(file)
    }

    ApiRoute --> Predictor
    Predictor --> ImageValidator
    Predictor --> Preprocessor
    Predictor --> ModelService
    Predictor --> ResultFormatter
```

---

## 19. Factory Pattern (creating detectors for comparison)

Useful if VeriPix compares FatFormer with a baseline such as UniFD.

```mermaid
classDiagram
    class Detector {
        <<interface>>
        +predict(tensor) float
    }
    class FatFormerDetector {
        +predict(tensor) float
    }
    class UniFDDetector {
        +predict(tensor) float
    }
    class DetectorFactory {
        +create(name) Detector
    }
    class Evaluator {
        +run(detector, dataset)
    }

    Detector <|.. FatFormerDetector
    Detector <|.. UniFDDetector
    DetectorFactory ..> Detector : creates
    Evaluator --> DetectorFactory
    Evaluator --> Detector
```

---

## 20. Repository Pattern (prediction history, later phase)

```mermaid
classDiagram
    class PredictionRepository {
        <<interface>>
        +save(result)
        +get_all() list
        +get_by_id(id) PredictionResult
    }
    class SqlitePredictionRepository {
        +save(result)
        +get_all() list
        +get_by_id(id) PredictionResult
    }
    class HistoryService {
        +record(result)
        +list_history() list
    }
    class PredictionResult {
        +label: str
        +fake_probability: float
        +processing_time_ms: float
    }

    PredictionRepository <|.. SqlitePredictionRepository
    HistoryService --> PredictionRepository
    PredictionRepository ..> PredictionResult
    SqlitePredictionRepository --> DataStore
    class DataStore {
        <<database>>
        SQLite file
    }
```

---

### Patterns deliberately not included
Observer, Command, Mediator, Broker, Peer-to-Peer, Publish-Subscribe, Shared Data and Map-Reduce are not used by VeriPix's design, so they are not drawn here. Add one only if the design actually uses it.

