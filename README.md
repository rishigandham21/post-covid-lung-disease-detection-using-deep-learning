# 🫁 Post-COVID Lung Disease Detection Using Deep Learning 💡

## 📖 About the Project

International healthcare systems are facing unprecedented challenges due to the **COVID-19 pandemic**, especially in detecting and managing its long-term effects.

👨‍⚕️ Survivors often develop:

- 🫁 **Lung Fibrosis**
- 🤒 **Pneumonia**
- 🎗️ **Lung Cancer**

However, factors like:
- ⚠️ Limited medical resources
- ⏱️ High patient workloads
- ❌ Subjective X-ray interpretation

...make diagnosis difficult and error-prone.

### ✅ Our Solution

This project presents an **AI-powered web application** that uses the 🧠 **Xception CNN model** to **classify chest X-ray images** into:

- `COVID-19`
- `Pneumonia`
- `Normal`

and also provides **precautionary health advice** based on the prediction.

---

## 🏗️ Tech Stack

| Component    | Technology               |
|--------------|---------------------------|
| 👨‍💻 Backend     | Python, Flask             |
| 🤖 Model       | Xception (Transfer Learning) |
| 📊 Visualization | Matplotlib, Seaborn        |
| 🖼️ Frontend     | HTML5, Bootstrap 5         |
| 🗂️ Preprocessing | OpenCV, NumPy, TensorFlow  |

---

## 📂 Dataset Structure

The dataset should be organized into the following directory structure:

| Split  | Class       | Folder Path                      |
|--------|-------------|----------------------------------|
| Train  | COVID-19    | `dataset/train/COVID/`           |
|        | Normal      | `dataset/train/Normal/`          |
|        | Pneumonia   | `dataset/train/Pneumonia/`       |
| Val    | COVID-19    | `dataset/val/COVID/`             |
|        | Normal      | `dataset/val/Normal/`            |
|        | Pneumonia   | `dataset/val/Pneumonia/`         |
| Test   | COVID-19    | `dataset/test/COVID/`            |
|        | Normal      | `dataset/test/Normal/`           |
|        | Pneumonia   | `dataset/test/Pneumonia/`        |

---

## 🧠 Model Overview

- 🔍 **Architecture**: Xception (Pretrained on ImageNet)
- ⚙️ **Input Size**: 224x224
- 🧪 **Output**: 3-class Softmax Layer
- 🏷️ **Classes**: COVID-19, Pneumonia, Normal
- 📉 **Loss Function**: Categorical Crossentropy
- ⚙️ **Optimizers**: Adam (with learning rate tuning)

---
