# 📂 Project File Map: Modular Clinical AI Architecture

To ensure clinical maintainability and academic clarity, the repository has been organized into functional tiers. 

---

### 🏛️ 1. `/core` (The Engine)
Contains the definitive model architectures and data loading pipelines.
- `clot_hybrid_v6.py`: Final CNN-Transformer-BiLSTM hybrid architecture.
- `v6_honest_dataloader.py`: Subject-wise, non-leaking data pipeline.

### 🚀 2. `/train` (Production Training)
Contains the training logic and hyper-parameter configurations.
- `train_hybrid_v6_final.py`: Option 1 balanced training loop (1:2.5:5 weights).
- `train_hybrid_v11.py`: Recovery loop (1:5:5 experimental weights).

### 🩺 3. `/eval` (Clinical Audits)
Scripts for performance verification and inference gating.
- `honest_audit_matrix.py`: Proves the **82.69% accuracy** and generates the Confusion Matrix.
- `inference_v11_threshold.py`: Implementation of the **P > 0.35** early-warning gate.
- `thesis_master_audit.py`: Generates the top-level precision/recall report.

### 🧪 4. `/preprocessing` & `/augmentation`
Advanced signal cleaning and class-balancing engines.
- `signal_processor_v12.py`: The Wavelet-Denoising (DWT) and Baseline Removal engine.
- `v12_spectral_audit.py`: Statistical proof of SNR improvement.
- `augment_warnings.py`: Time-Warping augmentation for the moderate stress tier.

### 📄 5. `/reports`
The definitive technical evidence for the dissertation.
- `COMPREHENSIVE_THESIS_PROJECT_REPORT.md`: The master 12-Phase synthesis.

### 📦 6. `/legacy` & `/trained_models`
- `/legacy`: Archive of 50+ intermediate versions (v4, v5) and debug scripts.
- `/trained_models`: The final verified weights (`clot_hybrid_v6_1_recovered.pth`).

---
**Lead AI Assistant**: Antigravity (DeepMind Advanced Agentic Coding)
**Sync Status**: Modular Architecture Verified
