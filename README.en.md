<div align="center">

# 🕯️ BrainDance | 流光 · 记 (LiuGuang · Ji)

> **流光 (Liúguāng)**: Flowing light — the transient beauty of moments slipping away
> **记 (Jì)**: Chronicles — recording and preserving what matters

**"The physical world is destined for disorder, but we rebuild eternity in the digital realm."**

**"This is the search engine for the physical world, a spatial visualization of the second brain."**

[English](README.en.md) | [简体中文](README.md)

### 🏆 A 3D Semantic Memory Engine for the Spatial Computing Era

#### An Anti-Entropy Engine for Human Memory

![Supabase](https://img.shields.io/badge/Supabase-Enabled-3ECF8E?logo=supabase&logoColor=white) ![Flutter](https://img.shields.io/badge/Flutter-Client-02569B?logo=flutter&logoColor=white) ![Python](https://img.shields.io/badge/Python-Worker-3776AB?logo=python&logoColor=white) ![License](https://img.shields.io/badge/license-MIT-blue)

</div>

## 📖 Project Overview

**BrainDance (流光 · 记)** is a **"Retrievable 3D Memory Library for Mobile"**.

Unlike traditional photo albums that can only capture 2D "images," BrainDance utilizes cutting-edge computer graphics technologies such as **3D Gaussian Splatting** to transform real-world physical spaces (your room, old streets about to be demolished, cherished collectibles) into digital assets with 1:1 high fidelity.

Furthermore, we integrate **Multimodal AI** and **RAG (Retrieval-Augmented Generation)** technologies to give these 3D scenes "semantics." You can **search the physical world** like using a search engine—ask it "Where did I leave my keys?" and the camera will automatically fly to and focus on that moment in space-time.

### Core Features

- **📷 Low-Cost Mobile Scanning**: Leverage mobile phones to capture video streams and pose data. Through extensive optimization and AI intervention, we've significantly improved the quality of 3DGS models generated from low-quality footage, greatly lowering the barrier to 3DGS modeling.
- **🔍 Spatial Semantic Retrieval (Spatial RAG)**: Combine multimodal large language models to understand scene content, enabling "Ctrl+F" search of the physical world.
- **⏳ Time Peeling**: Superimpose multi-dimensional time slices in the same coordinate system to achieve a visual experience from "now" back to "then."
- **☁️ Edge-Cloud Collaborative Rendering**: Mobile collection → Cloud high-performance computing → Mobile/XR lightweight viewing.

## 📜 Prologue: The War Against Entropy

**Physics** tells us that the ultimate fate of the universe is **entropy increase**. Houses deteriorate, objects break, order descends into chaos.

At the **biological** level, entropy increase manifests as **forgetting**. The decline of the hippocampus makes us forget the way home, the faces of loved ones. At the **sociological** level, entropy increase manifests as **demise**. Under the bulldozers of urban renewal, old streets, alleys, and their vibrant atmosphere will eventually turn to dust.

Existing technologies—2D photos and videos—are merely pale "screenshots" of reality. They lose depth, lose light and shadow, and lose the sense of space. They cannot resist forgetting because they are inherently flat.

**BrainDance (流光 · 记)** is more than just an app; it's a universal tool for humanity to fight against temporal entropy. Utilizing cutting-edge technologies like **3D Gaussian Splatting** and **Multimodal AI**, we attempt to capture the radiance field, **establish negative entropy in the digital world**, and leave behind a **spatial archive** that can transcend time for everyone and every city.

## 🌌 Value Coordinates: Micro, Macro, and Temporal

BrainDance's value architecture spans three dimensions, building a complete memory ecosystem from individuals to civilization:

### 1. The Micro Scale:

> **"Building a digital hippocampus for memories about to fade."**

- **Personal Witness (Spatial Journal)**:
  - **Space Folding**: When you graduate and leave a dorm you've lived in for 4 years, or move out of a rental filled with memories, one-click scanning lets you pack away the entire physical space. The building stays behind, but "home" travels with you.
- **Medical Assistance (The Cure)**:
  - **Nostalgia Therapy**: For **Alzheimer's** patients, VR/MR devices can transport them back to their childhood home. Photographs on the wall speak, meals on the table steam with warmth. This immersive familiarity offers comfort that medicine alone cannot provide.

### 2. The Macro Scale:

> **"A digital ark for the city, fighting against the fracture of civilization."**

- **Crowd-Sourced Archive**:
  - Cities grow, and they also disappear. BrainDance aggregates scan data from thousands of users to piece together a **3DGS map of the city**—preserving historic lanes, century-old shops, and ancient trees before they vanish.
- **Collective Memory**:
  - Even when physical entities are demolished and rebuilt, in BrainDance's parallel universe, those streets still exist—where neighbors knew each other's names, where the aroma of home cooking drifted through windows, where life unfolded at a human scale. Future generations won't read history through cold text, but will personally "walk into" history and touch the texture of civilization.

### 3. The Temporal Scale:

> **"The 'digital negative' of the spatial computing era."**

- **The Digital Negative**:
  - 2D video resolution is locked, and flat devices will inevitably become obsolete. But BrainDance records **light fields (radiance fields)**.
- **Future-Proof**:
  - Just as film movies can be remastered into 4K Blu-ray, data scanned today, in the era when Apple Vision Pro or Meta Quest become widespread, will deliver a sense of presence hundreds of times stronger than now. We are stockpiling **native spatial assets** for the next computing era.

## ⚡ Core Functions and Technical Philosophy

### Spatial RAG: Searching Reality Like Searching Text

We haven't just reconstructed "form," we've endowed "meaning." By integrating the visual understanding capabilities of **Multimodal LLMs**, BrainDance transforms unstructured 3D scenes into **searchable semantic databases**.

- **User Query**: "Where's that pocket watch my grandfather left me?"
- **System Action**: Semantic understanding → Spatial index matching → Automatic camera flyover → **Display the pocket watch deep in the drawer**.



### 🛠️ Technical Architecture and Implementation

This project adopts a **Supabase BaaS architecture**, implementing a fully automated workflow from mobile collection to cloud reconstruction.

The system consists of four parts, decoupled through **Supabase**:

1. **Client (Flutter)**:
   - Responsible for video capture and upload, supporting resumable uploads.
   - Direct connection to Supabase Storage/DB without middleware.
   - Monitor task progress through Realtime (supports Dynamic Island/Live Activities).
2. **Backend as a Service (Supabase)**: This project completely removes traditional middleware (Redis/MinIO/Go), building a fully Serverless architecture based on Supabase:
   - **PostgreSQL (The "Everything" Store)**:
     - **Vector DB**: Enable `pgvector` extension to store semantic vectors (Embeddings) of 3D scenes, implementing multimodal RAG retrieval.
     - **Job Queue**: Based on `processing_tasks` table and `FOR UPDATE SKIP LOCKED` mechanism, achieving high-performance, atomic task distribution queue (replacing Redis).
   - **Storage**:
     - **Bucket**: Unified management of raw video streams and trained 3D models (PLY/Splat), with public/private access controlled by Policy.
   - **Auth & Security**:
     - **RLS (Row Level Security)**: Database-native row-level security policies ensure users can only access their own 3D assets, achieving strict authentication with zero backend code.
   - **Realtime**:
     - **WebSocket**: Clients subscribe to database change stream (CDC), achieving millisecond-level task progress push and multi-end state synchronization (replacing frontend polling).
3. **Edge Functions (Deno)**:
   - **Serverless API**: Host lightweight business logic.
   - **Semantic Search**: Responsible for RAG semantic retrieval API, calling LLM Embedding API and performing vector matching, protecting API Keys from leakage.
4. **AI Worker (Python)**:
    - Pure compute nodes deployed on WSL/Linux GPU servers.
    - **Consumer**: Monitor database task queues, automatically pull videos or images.
    - **Training**: Run 3DGS/Nerfstudio training pipelines to generate PLY models.
    - **Single Image**: Support SAM3D-based single-image 3DGS generation without video.
    - **Understanding**: Call multimodal large models (Qwen-VL) for scene understanding and auto-tagging.



------

### 📂 Directory Structure

This project follows a **Monorepo** strategy, with all services hosted in the same repository and strict module isolation.

```
BrainDance/
├── ai_engine/            # [Python] Core Algorithm Engine (Worker)
│   ├── 3dgs/             #   - 3DGS Core Engine
│   │   ├── src/          #   - Source Code
│   │   │   ├── core/         #       - Pipeline base classes, factories, Worker
│   │   │   ├── pipelines/    #       - Pipeline implementations
│   │   │   │   ├── video_3dgs.py      #       - Video 3DGS Pipeline
│   │   │   │   └── single_image_sam3d.py  #       - Single Image SAM3D Pipeline
│   │   │   ├── modules/      #       - Functional Modules
│   │   │   │   ├── sam3d_engine/      #           - SAM3D Single Image Engine
│   │   │   │   ├── nerf_engine.py     #           - 3DGS Training Engine
│   │   │   │   ├── glomap_runner.py   #           - Pose Solving
│   │   │   │   └── knowledge_base.py  #           - RAG Knowledge Base
│   │   │   ├── libs/         #       - Embedded Dependencies
│   │   │   │   └── sam-3d-objects/    #           - SAM3D Inference Library
│   │   │   └── utils/        #       - Utility Functions
│   │   ├── tests/            #   - Test Scripts
│   │   ├── requirements.txt  #   - Python Dependencies
│   │   └── main.py           #   - Program Entry Point
│   ├── demo/              #   - Demo Scripts and Test Data
│   ├── models/            #   - AI Model Cache Directory
│   ├── rag/               #   - RAG Data Processing
│   └── log/               #   - Log Files
│
├── supabase/              # [BaaS] Cloud Infrastructure (Serverless)
│   ├── migrations/        #   - [SQL] Database Schema Change History
│   ├── seed.sql           #   - Initial Test Data
│   ├── config.toml        #   - Supabase Local Development Config
│   └── README.md          #   - ☁️ Backend Deployment Guide
│
├── docs/                  # [Doc] Project Documentation
│   ├── API_DOC.md         #   - API Interface Documentation
│   ├── BrainDance Project Collaboration Specification and Development Agreement (v1.0).md  #   - Development Specifications
│   ├── 待办/               #   - TODO Items
│   └── 技术报告/           #   - Technical Reports
│
└── README.md              #   - This File
```

> **Note**:
> - `app/` (Flutter Mobile Client) is under development and not yet included in this repository
> - `supabase/functions/` (Search Edge Functions) is under development

## 🚀 Quick Start

### Prerequisites

- **AI Engine**: NVIDIA GPU (CUDA 11.8+), Python 3.10+ (Conda recommended)
- **Infrastructure**: [Docker Desktop](https://www.docker.com/), [Supabase CLI](https://supabase.com/docs/guides/cli)
- **Client**: Flutter SDK (3.10+), Android Studio / Xcode

### Deployment Steps

This project supports **fully local deployment** and can run completely without cloud accounts.

#### 1. Start Infrastructure (Supabase Local)

One-click launch of database, storage buckets, authentication services, and Edge Functions based on Docker.



```bash
# 1. Enter project root directory
cd BrainDance

# 2. Start Supabase local services
supabase start

# 3. 🎉 Record the output API URL and Keys (Anon / Service Role)
#    - API URL: http://127.0.0.1:54321
#    - DB URL: postgresql://postgres:postgres@127.0.0.1:54322/postgres
#    - Studio: http://127.0.0.1:54323 (Web Console)
```

#### 2. Start Compute Engine (AI Worker)

Worker is responsible for monitoring local Supabase task queues and calling GPU for training.

```bash
cd ai_engine

# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment variables (copy example and fill in URL/Key from previous step)
cp .env.example .env
# ⚠️ Note: Worker must use SERVICE_ROLE_KEY to bypass RLS permissions

# 3. Start Worker
python src/worker.py
# Output: 🚀 [Worker] Connected to Supabase Local. Listening for tasks...
```

#### 3. Start Mobile Client (App)

Client connects directly to local Supabase gateway.

```bash
cd app

# 1. Configure connection
#    Edit lib/config.dart (or .env), fill in API URL and ANON_KEY

# 2. Run App
flutter run
```



## Sequence Diagram:

```mermaid
sequenceDiagram
    autonumber
    participant User as 👤 User
    participant App as 📱 Mobile Client (Flutter)
    participant Edge as ⚡ Edge Function
    participant Supabase as 🐘 Supabase (DB/Storage/Realtime)
    participant LLM as ☁️ LLM API (DashScope)
    participant Worker as ⚙️ Compute Node (Python Worker)

    Note over User, Supabase: === Phase 1: Direct Upload & Task Creation ===
    User->>App: 1. Click "New Memory", capture video
    App->>Supabase: 2. Direct upload video file (Storage Bucket: raw)
    Supabase-->>App: Return file path (Path)
    App->>Supabase: 3. Insert task record (Insert processing_tasks) <br/> {status: "pending", user_id: "..."}
    Supabase-->>App: Write successful

    par Real-time Feedback
        Supabase-->>App: 4. Realtime push: Status=Pending
        App-->>User: Display "Queuing..." (gray status)
    end

    Note over Supabase, Worker: === Phase 2: Async Consumption & Production ===
    loop Smart Polling
        Worker->>Supabase: 5. Claim task (RPC: pop_next_task) <br/> Using SKIP LOCKED
    end
    Supabase-->>Worker: Return task (Task A)

    Worker->>Supabase: 6. Download video (Storage)

    rect rgb(240, 248, 255)
        Note right of Worker: GPU-intensive Computing
        Worker->>Worker: 7a. 3DGS Training (Nerfstudio)
        Worker->>Worker: 7b. Scene Understanding & Auto-tagging (Qwen-VL)
    end

    Worker->>Supabase: 8. Upload .ply model (Storage Bucket: output)
    Worker->>Supabase: 9. Asset entry (Insert model_assets) <br/> Including Embedding vector (pgvector)
    Worker->>Supabase: 10. Update task status (Update status="completed")

    Supabase-->>App: 11. Realtime push: Status=Completed
    App-->>User: Dynamic Island/Notification pops up "Memory Construction Complete"

    Note over User, Worker: === Phase 3: Browse & RAG Search ===
    User->>App: 12. View "My Models"
    App->>Supabase: Query asset table (Select model_assets) <br/> RLS auto-filter
    Supabase-->>App: Return list (including .ply download links)
    App->>App: 13. Load 3D view (Flutter rendering)

    User->>App: 14. Search "Where are my keys"
    App->>Edge: 15. Call search API (POST /functions/v1/search)

    rect rgb(255, 250, 240)
        Note right of Edge: Serverless Semantic Search
        Edge->>LLM: 16. Get text vector (Embedding)
        LLM-->>Edge: Return Vector
        Edge->>Supabase: 17. Vector matching (RPC: match_model_assets) <br/> pgvector similarity calculation
        Supabase-->>Edge: Return Top-K results
    end

    Edge-->>App: Return JSON result
    App-->>User: Display search results and locate viewpoint
```

------



### 📚 References & Acknowledgements

This project is an exploration standing on the shoulders of giants. Core algorithms and rendering capabilities heavily draw from and integrate the following excellent open-source projects, to whom we express our gratitude:

#### Core Algorithms

- **[nerfstudio](https://github.com/nerfstudio-project/nerfstudio)**: Provides the most modular NeRF/3DGS training framework. Our training pipeline is modified based on the `splatfacto` model.
- **[gsplat](https://github.com/nerfstudio-project/gsplat)**: Ultra-fast CUDA rasterization backend, providing performance assurance for cloud training.
- **[gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)**: Inria's original paper implementation, laying the theoretical foundation.
- **[SAM3D](https://github.com/ ScreenVerse/sam-3d-objects)**: Single-image 3DGS generation framework, supporting high-quality 3D model reconstruction from a single photo.
- **[SHARP](https://github.com/apple/ml-sharp)**: Apple's high-quality single-image 3DGS generation model, directly predicting Gaussian splatting parameters through neural networks.

#### Infrastructure & AI

- **[Supabase](https://github.com/supabase/supabase)**: The soul of this project. Provides out-of-the-box Auth, Storage, and Realtime capabilities, allowing us to focus on 3D business logic.
- **[pgvector](https://github.com/pgvector/pgvector)**: PostgreSQL vector extension, replacing ChromaDB and providing high-performance RAG retrieval capabilities for this project.
- **[Qwen-VL](https://github.com/QwenLM/Qwen-VL)**: Powerful multimodal large model that gives 3D scenes the ability to "be understood" (auto-tagging and description).

#### Rendering & Viewer

- **[GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D)**: Three.js-based web viewer, inspiration for our mobile WebView rendering.
- **[antimatter15/splat](https://github.com/antimatter15/splat)**: Another excellent WebGL implementation, providing early conceptual references.





We believe technology shouldn't just be cold parameter competition. **The best technology is to let Dasein no longer be lonely, to let transience become eternal.**

<div align="center">


<sub>Made with ❤️ by the BrainDance Team: 烫锟斤拷烫. Dedicated to everyone fighting against entropy.</sub>

</div>
