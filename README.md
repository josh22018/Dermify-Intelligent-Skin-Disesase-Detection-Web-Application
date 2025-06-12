# Dermify-Intelligent-Skin-Disesase-Detection-Web-Application

Dermify is a deep learning-based web application designed for the **automated detection and classification of dermatological conditions**. It leverages the power of Vision Transformers (ViT) and the self-supervised **DinoV2** model to achieve **state-of-the-art accuracy of 96.48%** across a curated dataset of 31 skin disease classes.

## 🧠 Project Highlights

* 🚀 **High Accuracy**: Achieves 96.48% accuracy using advanced transformer models.
* 🔍 **Explainability**: Incorporates **GradCAM** and **SHAP** for interpretable AI decisions.
* 🌐 **Web Interface**: Built using **Flask**, the app provides a user-friendly interface for uploading images and viewing predictions.
* 💬 **AI Chatbot Integration**: Integrated with **Groq’s LLaMA 3.3** chatbot for disease-specific Q\&A and insights.

---

## 📊 Model Architecture

Dermify uses a **Vision Transformer (ViT)** combined with the **DinoV2 self-supervised learning model**, enabling it to:

* Capture **both local and global features** of dermatological images.
* Improve classification performance significantly over traditional CNNs.
* Generalize well across diverse skin conditions in the dataset.

---

## 🔎 Explainable AI (XAI)

To ensure **transparency and clinical trust**, Dermify integrates:

* **GradCAM**: Visual heatmaps highlighting important image regions.
* **SHAP**: Feature importance explanations for each prediction.

These tools aid both patients and healthcare professionals in understanding model decisions.

---

## 🌐 Web Application Features

* 📤 **Image Upload**: Drag-and-drop or select images from your device.
* ⚡ **Real-Time Predictions**: Instant classification with confidence scores.
* 🗺️ **Visual Explanations**: Overlay of GradCAM heatmaps on uploaded images.
* 🧑‍⚕️ **Chatbot Interface**: Ask the Groq-powered LLaMA 3.3 chatbot for insights on diagnosed diseases.

---

## 🛠️ Tech Stack

* **Frontend**: HTML/CSS, JavaScript
* **Backend**: Python, Flask
* **ML Models**: Vision Transformer (ViT), DinoV2
* **Explainability**: GradCAM, SHAP
* **Chatbot**: Groq LLaMA 3.3 Integration
* **Deployment**: Flask Web Server

---

## 🗂️ Dataset

Dermify was trained on a **curated 31-class skin disease dataset**, featuring a diverse range of dermatological conditions with high-quality image samples.

---

