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
- Dataset preparation and image preprocessing  
- CNN model development using PyTorch  
- Training a supervised baseline model  
- Adding semi-supervised learning with pseudo-labeling  
- Filtering low-confidence predictions  
- Model evaluation and performance comparison  
- Visualization of results using metrics and plots  

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
1. Clone the repository  
2. Install dependencies using `pip install -r requirements.txt`  
3. Place dataset in `COVID-19_Radiography_Dataset/`  
4. Run `python chest_xray_ssl_training.py`
