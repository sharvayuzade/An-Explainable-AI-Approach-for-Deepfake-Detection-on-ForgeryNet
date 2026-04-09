# An Explainable AI Approach for Deepfake Detection on ForgeryNet

A deep-learning pipeline that detects image forgeries using a fine-tuned **ResNet-18** model trained on the **CASIA2** dataset, and explains *where* the forgery occurs with **Grad-CAM** heatmaps and an **LLM-generated** natural-language summary.

---

##  Project Flow Diagram

```mermaid
flowchart TD
    A([ Input: CASIA2 Dataset\nAuthentic & Tampered Images]) --> B

    subgraph Preprocessing[" Data Preprocessing"]
        B[List & Split Images\nAuthentic vs. Tampered]
        B --> C[Resize to 224×224\nImageNet Normalisation]
        C --> D[PyTorch DataLoader\nBatch Size = 12]
    end

    D --> E

    subgraph Training[" Model Training"]
        E[Pretrained ResNet-18\nFreeze All Layers]
        E --> F[Unfreeze layer4 + FC Head\nBinary Output: Authentic / Tampered]
        F --> G[Fine-tune with\nCrossEntropyLoss & Adam Optimiser\nEpochs = 2  LR = 1e-4]
        G --> H{Validation\nAccuracy Improved?}
        H -- Yes --> I[Save Best Checkpoint\ncasia_resnet18_checkpoint.pt]
        H -- No --> G
    end

    I --> J

    subgraph Inference[" Inference on New Image"]
        J[Load Checkpoint\n+ Inference Transform]
        J --> K[Forward Pass → Softmax\nClass Probabilities]
        K --> L{Prediction}
        L -- Authentic --> M1([ Authentic])
        L -- Tampered --> M2([ Tampered])
    end

    K --> N

    subgraph XAI["🌡️ Explainability — Grad-CAM"]
        N[Register Forward & Backward Hooks\non ResNet-18 layer4]
        N --> O[Compute Gradient-Weighted\nActivation Map]
        O --> P[Resize CAM → Original Image Size\nApply Jet Colormap Overlay]
    end

    P --> Q

    subgraph LLM[" LLM Natural-Language Explanation"]
        Q[Send Prediction + Confidence\nto Ollama LLaMA 3.2:3b]
        Q --> R[Generate Plain-English\nForgery Summary]
    end

    P --> S
    R --> S

    subgraph Frontend[" Frontend / UI"]
        S[Gradio Interface\nUpload Image]
        S --> T[Display: Prediction · Confidence\nGrad-CAM Heatmap · LLM Summary]
    end

    T --> U([ Final Output\nExplainable Forgery Detection Result])

    style Preprocessing fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a
    style Training      fill:#dcfce7,stroke:#16a34a,color:#14532d
    style Inference     fill:#fef9c3,stroke:#ca8a04,color:#713f12
    style XAI           fill:#fce7f3,stroke:#db2777,color:#831843
    style LLM           fill:#ede9fe,stroke:#7c3aed,color:#3b0764
    style Frontend      fill:#ffedd5,stroke:#ea580c,color:#7c2d12
```

---

##  Key Components

| Component | Details |
|-----------|---------|
| **Dataset** | [CASIA2](https://github.com/namtpham/casia2groundtruth) – Authentic & Tampered image pairs |
| **Model** | ResNet-18 (ImageNet pre-trained), fine-tuned for binary forgery classification |
| **XAI Method** | Grad-CAM on `layer4` — highlights manipulated regions |
| **LLM Explanation** | Ollama · LLaMA 3.2:3b — generates plain-English forgery description |
| **UI** | Gradio web interface embedded in `xai_frontend.html` |
| **Checkpoint** | `casia_resnet18_checkpoint.pt` / `casia_resnet18_forgery.pth` |

---

##  Quick Start

### 1. Install Dependencies
```bash
pip install torch torchvision scikit-learn matplotlib pillow gradio opencv-python
```

### 2. (Optional) Start Ollama for LLM Explanations
```bash
ollama serve
ollama pull llama3.2:3b
```

### 3. Run the Notebook
Open `implementation.ipynb` in Jupyter and run all cells top-to-bottom.  
The last cell launches the **Gradio** app at `http://127.0.0.1:7860`.

### 4. Open the Frontend
Open `xai_frontend.html` in your browser — it embeds the Gradio app in a clean UI.

---

##  Repository Structure

```
.
├── implementation.ipynb          # Full training + inference + XAI + Gradio pipeline
├── xai_frontend.html             # Standalone HTML frontend for the Gradio app
├── casia_resnet18_checkpoint.pt  # Saved model checkpoint (state dict + metadata)
├── casia_resnet18_forgery.pth    # Alternative saved model weights
├── public_images_predictions.csv # Predictions on public test images
└── README.md                     # This file
```

---

##  Pipeline Summary

1. **Data Loading** — Scans `CASIA2/Au` (authentic) and `CASIA2/Tp` (tampered) folders.
2. **Preprocessing** — Resize → Normalize (ImageNet stats) → DataLoader.
3. **Fine-Tuning** — Freeze backbone, unfreeze `layer4` + FC head, train 2 epochs.
4. **Checkpointing** — Best validation-accuracy model is saved.
5. **Inference** — Load checkpoint, run image through model, get class + confidence.
6. **Grad-CAM** — Hooks on `layer4` produce a saliency heatmap over the image.
7. **LLM Summary** — Prediction + confidence sent to LLaMA 3.2 for a plain-English explanation.
8. **Gradio UI** — Single-page interface: upload → see prediction, heatmap, and text explanation.
