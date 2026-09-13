# 🩻 Chest X-Ray Classification with Semi-Supervised Learning

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/vision/stable/index.html)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

---

### 💡 Core Problem & Approach

Acquiring expert-annotated medical datasets is prohibitively expensive and time-consuming. This system addresses labeled data scarcity by initializing a supervised backbone on a minimal annotated subset, predicting soft target distributions for unlabeled radiographs, filtering out low-confidence predictions, and retraining the model on the augmented dataset.

Labeled Images ──> Train Baseline CNN ──> Predict Unlabeled Images ──> Filter (Confidence > Threshold) ──> Train Combined Set ──> Final Retrained CNN

---

### 📌 Project Overview
This project builds a deep learning system to classify chest X-ray images into medical categories such as COVID-19, pneumonia, and normal cases. It compares supervised learning with semi-supervised learning to show how unlabeled medical data can improve performance when labeled data is limited.

---

### 🛠️ Technologies Used

* **Core & Frameworks:** Python, PyTorch, Torchvision
* **Data & Image Processing:** NumPy, Pillow (PIL), Torchvision Transforms
* **Model Evaluation & Viz:** Scikit-learn, Matplotlib, Seaborn

---

### 📌 Key Features

* **Multi-Class Radiograph Classification:** Categorizes chest X-rays across four target classes: *COVID-19*, *Viral Pneumonia*, *Lung Opacity*, and *Normal*.
* **Supervised Baseline Benchmark:** Establishes performance bounds using standard cross-entropy optimization on strictly labeled data splits.
* **Confidence-Thresholded Pseudo-Labeling:** Filters pseudo-labeled samples via Softmax probability thresholds to prevent error propagation and noisy gradients during retraining.
* **Comprehensive Evaluation Metrics:** Evaluates diagnostic performance via multiclass ROC-AUC curves, macro/micro F1-scores, precision/recall, and confusion matrices.

---

### ⚙️ The Pipeline

1. **Dataset Structuring & Data Augmentation**
   * Preprocessed the *COVID-19 Radiography Dataset*, scaling images to unified spatial dimensions.
   * Standardized pixel intensity channels and applied geometric transformations (random rotations, horizontal flips) to prevent overfitting.
   * Split data to explicitly simulate realistic semi-supervised conditions (small labeled anchor set, large unlabeled pool).

2. **Supervised Baseline Initialization**
   * Built and trained a deep Convolutional Neural Network backbone using labeled anchor images.
   * Optimized using Cross-Entropy Loss and Adam optimizer to set comparative baseline metrics.

3. **Semi-Supervised Generation & Filtering**
   * Executed inference over the unlabeled dataset split to generate predicted class probability distributions.
   * Enforced a strict confidence cut-off threshold ($\tau$), discarding uncertain outputs and retaining only high-probability pseudo-labels.

4. **Iterative Model Retraining**
   * Concatenated ground-truth labeled samples with the newly accepted pseudo-labeled samples into an expanded training set.
   * Retrained the CNN architecture on the combined distribution, boosting feature learning from unlabeled image patterns.

5. **Diagnostic Performance Benchmarking**
   * Evaluated model generalization against a dedicated test set.
   * Generated class-wise confusion matrices and ROC curves to measure performance gains between supervised vs. semi-supervised training.

---

### 📚 What I Learned
* **Label Efficiency:** Semi-supervised pseudo-labeling achieved performance comparable to fully supervised baselines while utilizing significantly fewer manual annotations.
* **Threshold Sensitivity:** Highlighted the critical trade-off between pseudo-label volume and noise introduction; high confidence thresholds are vital to preventing target drift.
* **Diagnostic Imbalance Handling:** Demonstrated the necessity of robust evaluation metrics (F1-Score, ROC-AUC) over raw accuracy when handling non-uniform clinical class distributions.

---

### 🚀 How Can It Be Improved?

* **Backbone Upgrades:** Transition from custom CNN architectures to transfer learning backbones (ResNet-50, EfficientNet, or Vision Transformers).
* **Advanced Semi-Supervised Methods:** Integrate state-of-the-art semi-supervised frameworks such as **FixMatch**, **MixMatch**, or **FlexMatch**.
* **Model Explainability:** Integrate **Grad-CAM** heatmaps to visualize activation regions and validate clinical focus areas.
* **Production Deployment:** Wrap the inference engine into a lightweight RESTful microservice using Flask/FastAPI and Docker.

---

## ▶️ Running the Project

### 1. Clone the repository

`git clone https://github.com/your-username/your-repo-name.git`

`cd your-repo-name`

### 2. Install dependencies

`pip install -r requirements.txt`  

### 3. Download the dataset

Download the dataset from Kaggle:
https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

Extract it and ensure the folder is named:
`COVID-19_Radiography_Dataset/`  

### 4. Folder structure

Make sure your project follows the same folder structure as shown in the repository above.

### 5. Run the project

`python Chest_Xray_SSL_Training.py`
