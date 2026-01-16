# Diabetic Retinopathy Detection – End-to-End ML System

This repository presents an end to end diabetic retinopathy (DR) detection system, including deataset prepparation, preprocessing, model training, evaluation, and Dockerized deployment as FastAPI iference service.

The goal of this project is to demonstrate a complete reproducible medical image anaylsis pipeline, from raw data to deployable machine learning service

# Project Highlights

- End-to-end pipeline: dataset preparation → preprocessing → training → evaluation → inference
- Fundus image-based DR / NO DR classification
- PyTorch deep learning model
- FastAPI-based inference service
- Dockerized deployment for reproducibility
- REST API with image upload support

# Dataset

This project is based on the EyePACS diebetic retinopathy dataset, a large-scale fundus image dataset commonly used for DR classification research.

- Source: https://www.kaggle.com/competitions/diabetic-retinopathy-detection/overview
- Modality: Retinal fundus images
- Labels: Diabetic retinopathy severity levels:
- **0 – No DR**  
- **1 – Mild**  
- **2 – Moderate**  
- **3 – Severe**  
- **4 – Proliferative DR**
- 
# Data Preparation & Preprocessing
## Dataset Selection & Conversion

The **EyePACS dataset** was used as the primary data source.  
Images were reorganized into an **EyeQ-style directory structure** to ensure compatibility with quality-based filtering and preprocessing steps.

Image quality labels used for filtering and preprocessing were obtained from the **EyeQ project**:

- **EyeQ GitHub Repository:** https://github.com/HzFu/EyeQ  

The **EyeQ image quality annotations** were used to identify low-quality fundus images and perform quality-based dataset selection.

### Label Mapping

Labels were mapped for binary classification:

- **DR** – Diabetic Retinopathy  
- **NO DR** – Non-Diabetic Retinopathy  

---
- **Original Label 0** → Mapped to **0 (No DR)**
- **Original Labels 1 and 2** → Excluded from the dataset
- **Original Labels 3 and 4** → Mapped to **1 (DR)**

As a result, the final dataset contains two classes:

- **0: No Diabetic Retinopathy**
- **1: Diabetic Retinopathy**

## Image Quality Handling

Low-quality or invalid fundus images were identified and filtered.  
This step helps to:

- Reduce noise  
- Remove corrupted samples  
- Improve overall model robustness  

---

## Image Preprocessing

The following preprocessing steps were applied:

- Center cropping and padding to preserve the retinal region  
- Image resizing to match CNN input requirements  
- Pixel normalization for deep learning training  

---

## Dataset Splitting

- **80% Training Set**
- **20% Test Set**
- Class imbalance handled during training using **class weighting**
  
    
# Model Overview
- Task: Binary classification (DR vs NO DR)
- Input: Fundus retinal images
- Architecture: ResNet50
- Framework: PyTorch
- Output: Prediction label and probability score

# Training & Evaluation
- Training logic implemented in `engine/train.py`
- Evaluation performed using validation accuracy and loss
- Configuration managed via YAML files under `configs/`
- Best-performing model weights are saved for deployment

# Deployment & Inference
-The trained model is deployed as a **FastAPI service**, packaged using Docker to
ensure full reproducibility across systems

# Build Docker Image

```bash
docker build -t dr-detection-api .
docker run -p 8000:8000 dr-detection-api
```

After running the container, open your browser:

```
http://localhost:8000
```




