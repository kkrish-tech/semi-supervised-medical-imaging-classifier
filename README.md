# 🩻 Chest X-Ray Classification with Semi-Supervised Learning

## 📌 Project Overview
This project builds a deep learning system to classify chest X-ray images into medical categories such as COVID-19, pneumonia, and normal cases. It compares supervised learning with semi-supervised learning to show how unlabeled medical data can improve performance when labeled data is limited.

---

## 🛠️ Technologies Used
- Python  
- PyTorch  
- TorchVision  
- Scikit-learn  
- NumPy  
- Matplotlib  
- Pillow (PIL)  

---

## ✨ Features
- Chest X-ray image classification (COVID-19, Pneumonia, Normal, Lung Opacity)  
- CNN-based deep learning model  
- Supervised baseline model  
- Semi-supervised learning using pseudo-labeling  
- Confidence-based filtering for unlabeled data  
- Evaluation using Accuracy, F1-score, ROC-AUC, and Confusion Matrix  

---

## ⚙️ The Process
1. Dataset Preparation  
• Loaded the COVID-19 Radiography Dataset  
• Organized chest X-ray images into labeled categories (COVID-19, Normal, Pneumonia, Lung Opacity)  
• Sampled a subset of images to simulate a limited-data scenario  

2. Image Preprocessing  
• Resized all images to a fixed resolution  
• Normalized pixel values for consistent CNN input  
• Applied basic augmentation to improve generalization  

3. Supervised Model Training  
• Built a Convolutional Neural Network (CNN) using PyTorch  
• Trained the model using only labeled data  
• Established baseline performance for comparison  

4. Semi-Supervised Setup  
• Split training data into labeled and unlabeled sets  
• Used the trained model to generate pseudo-labels for unlabeled data  

5. Pseudo-Label Filtering  
• Applied confidence thresholding to remove low-confidence predictions  
• Kept only reliable pseudo-labels for training  

6. Model Training with Unlabeled Data  
• Combined labeled data with high-confidence pseudo-labels  
• Retrained the CNN to improve performance using additional data  

7. Evaluation & Comparison  
• Evaluated models using accuracy, F1-score, confusion matrix, and ROC curves  
• Compared supervised vs semi-supervised performance  

8. Visualization of Results  
• Generated confusion matrices for class-wise performance  
• Plotted ROC curves to analyze classification quality  

---

## 📚 What I Learned
- Semi-supervised learning improves performance with limited labels  
- Practical CNN implementation in PyTorch  
- Importance of preprocessing in medical imaging  
- Handling class imbalance in datasets  
- Evaluation techniques for classification models  
- Trade-offs between supervised and semi-supervised approaches  

---

## 🚀 How Can It Be Improved?
- Use pretrained models like ResNet or EfficientNet  
- Improve pseudo-labeling strategies (FixMatch, MixMatch)  
- Add cross-validation for better evaluation  
- Hyperparameter tuning for better performance  
- Deploy as a web application (Flask/FastAPI)  
- Add Grad-CAM for model interpretability  

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
