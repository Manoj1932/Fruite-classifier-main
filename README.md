# 🍎 Fruit Classifier Project

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python) 
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange?logo=tensorflow) 
![License](https://img.shields.io/badge/License-MIT-green)

---

**Description:**  
This project classifies fruit images using a Convolutional Neural Network (CNN). It has two main components:

1. **Model Training** – Train an improved CNN on fruit datasets.  
2. **GUI + Voice Prediction** – Upload fruit images, predict their class, and announce predictions using text-to-speech.  

---

## 📋 Table of Contents

- [Features](#-features)  
- [Technologies Used](#-technologies_used)  
- [Installation](#-installation)  
- [Usage](#-usage)  
- [Training the Model](#-training_the_model)  
- [Screenshots](#-screenshots)  
- [Requirements](#-requirements)  
- [Contributing](#-contributing)  
- [License](#-license)  
- [Contact](#-contact)  

---

## ✨ Features 

- Train an improved CNN for fruit classification.  
- Data augmentation for robust training.  
- Upload fruit images through a user-friendly GUI.  
- Predict fruit type with confidence.  
- Text-to-speech voice output for predictions.  

---

## 🛠️ Technologies_Used

- Python 3.x  
- TensorFlow / Keras (CNN model)  
- Tkinter (GUI)  
- PIL / Pillow (image processing)  
- NumPy  
- pyttsx3 (text-to-speech)  
- Matplotlib (accuracy/loss plots)  

---

## ⚡ Installation

1. **Clone the repository:**

```bash
git clone https://github.com/jaddumohankishore/fruit-classifier.git
cd fruit-classifier
```

2. **Create and activate a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3.**Install dependencies:**

```bash
pip install -r requirements.txt
```

## 🎮 Usage
**GUI + Voice Prediction**

```bash

python fruit_classifier_gui_voice.py
```

**1.** Click **Upload Fruit Image** and select an image.

**2.** The GUI will display the image, predict the fruit type, and announce the prediction.

## Training_the_Model

**1.Organize your dataset:**
```bash
Fruit_classification/
    train/
        apple/
        banana/
        orange/
    test/
        apple/
        banana/
        orange/
```

**2.Run the training script:**
```bash
python improved_fruit_classifier.py
```

**3.The model will train with data augmentation and save the best models as:**

best_fruit_model.h5 and final_fruit_classifier.py

**4.Training plots (accuracy & loss) will be displayed using Matplotlib.**

## 🖼️ Screenshots

(Add screenshots of GUI, sample predictions, and training plots here)

## 📦 Requirements

requirements.txt:
```bash
tensorflow==2.14.0
numpy==1.25.2
Pillow==10.1.0
pyttsx3==2.90
matplotlib==3.8.0
```

Tkinter is included with Python; no need to install separately.

## 🤝 Contributing

Contributions are welcome!

**1.Fork the repository**
**2.Create a branch:**
```bash
git checkout -b feature/your-feature
```

**3.Commit changes:**
```bash
git commit -m 'Add feature'
```

**4.Push:**
```bash
git push origin feature/your-feature
```

**5.Open a Pull Request**

## 📝 License

Distributed under the MIT License. See LICENSE for details.

## 📬 Contact

**Jaddu Mohan Kishore –** jaddumohankishore@gmail.com

**Project Link:** https://github.com/jaddumohankishore/Fruit-classifier
