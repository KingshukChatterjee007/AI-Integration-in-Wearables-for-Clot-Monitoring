# Multimodal AI for Wearable Blood Clot Monitoring: A Bayesian Transformer Approach

**Idea Interpreter:** Md Azlan

**Lead Researcher:** Kingshuk Chatterjee  & Kanishk

**Project Phase:** v4.0 Balanced Production Optimization  
**Date:** March 23, 2026

---

## 1. Executive Abstract
This research details the development of a clinical-grade AI system for the continuous monitoring of thrombosis risk (blood clots) via non-invasive wearable sensors. By evolving from traditional machine learning (Phase 1) to a **Bayesian Feature-Space Transformer (v4)**, the project successfully solved the **"Accuracy Paradox."**

The current v4 architecture utilizes **Conditional Tabular GANs (CTGAN)** for minority class synthesis and **Monte Carlo (MC) Dropout** for uncertainty quantification. The final model achieves a **58% Critical Precision** and **52% Critical Recall** on strictly unseen patient populations, with a **54.1% reduction in model size** via INT8 quantization for edge-device deployment.

---

## 2. The Architectural Evolution: From 0% to 58%

### **Phase 1: The Baseline (The Accuracy Paradox)**
Initial benchmarking evaluated 11 algorithms (XGBoost, CatBoost, SVM, etc.) on 5,612 samples. 
* **The Failure:** While XGBoost yielded 84.38% Accuracy, it suffered from 0% Critical Recall. 
* **The Discovery:** We identified Data Leakage in 9 derived features (e.g., `composite_risk_score`). Removing these revealed the true model capability.

### **Phase 4: The SWAT Tier (The Current Standard)**
We transitioned to a Multimodal FT-Transformer to capture the temporal interplay between cardiovascular signals (BVP) and sympathetic nervous system volatility (EDA).

| Metric | Phase 1 (XGBoost) | Phase 4 (FT-Transformer) |
| :--- | :--- | :--- |
| **Model Type** | Gradient Boosted Trees | **Bayesian FT-Transformer** |
| **Balanced Samples** | 5,612 (Imbalanced) | **18,369 (Balanced via CTGAN)** |
| **Critical Precision** | 0.0% | **58.0% (Targeting 75% SWAT)** |
| **Safety Logic** | Blind Prediction | **Bayesian Safety Gate ($MI > 0.4$)** |

---

## 3. Mathematical & Algorithmic Framework

### **A. Multimodal Feature Tokenization**
The v4 Transformer projects each sensor reading into a high-dimensional embedding space, treating physiological states as "medical semantics."

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

This allows the model to dynamically weigh **EDA (Stress)** spikes higher when **BVP (Blood Volume)** variability drops.

### **B. Generative Minority Oversampling (CTGAN)**
To overcome the scarcity of "Critical" events, we implemented a **Conditional Tabular GAN**.
* **Method:** Learned the 154-dimensional probability distribution of confirmed thrombosis events.
* **Result:** Generated **5,000 synthetic high-fidelity samples** (Quality Score: 0.88).

### **C. Bayesian Uncertainty Quantification**
We run **$T=50$ stochastic forward passes** (MC-Dropout) during every inference to calculate **Mutual Information ($MI$)**.

$$MI = -\sum_{c=1}^{C} \hat{p}_c \ln(\hat{p}_c) - \frac{1}{T} \sum_{t=1}^{T} \sum_{c=1}^{C} p_{c,t} \ln(p_{c,t})$$

* **Clinical Gate:** If $MI > 0.4$, the system triggers `Status: UNCERTAIN`.

---

## 4. Repository & File System Deep-Dive
* **`integrated-scripts/`**
    * `ctgan_balancing.py`: The generative engine.
    * `focal_loss.py`: Custom PyTorch loss function.
    * `mc_dropout_inference.py`: The Bayesian engine.
    * `temporal_features.py`: Calculates Velocity (Rate of Change).
* **`trained_models/`**
    * `clot_transformer_v4.onnx`: The edge-optimized binary (418KB).

---

## 5. How to Run the Production Pipeline

**Step 1: Balance the Data** Execute the CTGAN generative engine to handle class imbalance by synthesizing 5,000 high-fidelity "Critical Risk" samples.
`bash
python integrated-scripts/ctgan_balancing.py

##Step 2: Train the "SWAT" Transformer Train the FT-Transformer using Weighted Focal Loss and temporal features to achieve the 58% precision threshold.

Bash
python integrated-scripts/train_optimized_v4.py

##Step 3: Bayesian Inference Run the inference engine with MC-Dropout to calculate both the risk category and the Mutual Information (uncertainty) score.

Bash
python integrated-scripts/mc_dropout_inference.py --input sample_sensor_data.csv

##6. Conclusion
The v4 Balanced Transformer represents a significant shift from "score-chasing" accuracy to trustworthy medical engineering. By solving extreme class imbalance with GANs and implementing a Bayesian Safety Gate, this system provides a robust, self-aware framework ready for pilot clinical trials and edge deployment in diverse, unseen patient populations.
