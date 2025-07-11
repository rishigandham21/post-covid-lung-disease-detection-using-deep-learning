# 🫁 Post-COVID Lung Disease Detection using Deep Learning 🧠📸

## 📖 About the Project

International healthcare systems are facing unprecedented challenges due to the **COVID-19 pandemic**, especially in managing its **long-term respiratory effects**. Survivors often develop complications such as:

- 🫁 **Lung Fibrosis**
- 🤒 **Pneumonia**
- 🎗️ **Lung Cancer**

These complications require **early and accurate detection** to ensure timely treatment. However, **resource constraints**, a **shortage of radiologists**, and the **subjective nature** of manual X-ray interpretation result in diagnostic delays and errors.

### 💡 Our Solution

This project introduces a **deep learning-based diagnostic system** powered by the 🧠 **Xception model**, a high-performance **Convolutional Neural Network (CNN)** designed for medical image analysis. Key benefits include:

- ✅ **High Accuracy** in classifying COVID-related lung conditions  
- ⚡ **Fast and Scalable** solution for high/low-resource environments  
- 🌍 **Telemedicine Ready** — remote diagnosis through chest X-rays  
- 🤖 **Reduces Radiologist Workload** with AI-assisted support

---

## 🏗️ Tech Stack

| Tool | Purpose |
|------|---------|
| TensorFlow/Keras 🧠 | Model Building & Training |
| Xception Model 🧬 | Transfer Learning Backbone |
| Matplotlib & Seaborn 📊 | Visualization |
| NumPy 🧮 | Numerical Operations |
| ImageDataGenerator 🖼️ | Data Augmentation & Preprocessing |

---

## 📂 Dataset Structure

The project expects a dataset structured as:

## 🗂️ Dataset Structure

| Set      | Class      | Folder Path                      |
|----------|------------|----------------------------------|
| Train    | COVID      | `dataset/train/COVID/`           |
|          | Normal     | `dataset/train/Normal/`          |
|          | Pneumonia  | `dataset/train/Pneumonia/`       |
|----------|------------|----------------------------------|
| Val      | COVID      | `dataset/val/COVID/`             |
|          | Normal     | `dataset/val/Normal/`            |
|          | Pneumonia  | `dataset/val/Pneumonia/`         |
|----------|------------|----------------------------------|
| Test     | COVID      | `dataset/test/COVID/`            |
|          | Normal     | `dataset/test/Normal/`           |
|          | Pneumonia  | `dataset/test/Pneumonia/`        |


