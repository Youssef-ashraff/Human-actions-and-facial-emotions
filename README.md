# Human Action & Facial Emotion Recognition System

## 📌 Overview
This project is a dual-stream computer vision system capable of performing **real-time Human Action Recognition** and **Facial Emotion Detection** simultaneously. It integrates a temporal deep learning model (CNN-LSTM) for analyzing action sequences with a specialized CNN for facial expression analysis.

The system is built with **TensorFlow/Keras** and **OpenCV**, featuring a graphical input selection menu (Tkinter) and a real-time visualization interface.

## 🚀 Key Features
* **Hybrid Recognition Pipeline:** Runs two independent deep learning models concurrently to analyze both body language (actions) and facial cues (emotions).
* **Action Recognition (HAR):** Utilizes a **Time-Distributed MobileNetV2 + LSTM** architecture to classify complex actions from video sequences (trained on UCF101).
* **Emotion Recognition (FER):** Uses a fine-tuned **ResNet50V2** with a custom grayscale-to-RGB adaptation layer to classify 7 emotions (trained on FER-2013).
* **Smart Inference:** Includes a "Warm-up" sequence to initialize GPU tensors, preventing latency spikes during the first few seconds of runtime.
* **Interactive UI:**
    * **Input Selector:** Tkinter-based GUI to choose between Live Webcam or Video File.
    * **Live Overlay:** Real-time bounding boxes and confidence scores rendered via OpenCV with a custom "Neon" aesthetic.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Deep Learning:** TensorFlow, Keras
* **Computer Vision:** OpenCV (`cv2`)
* **Data Processing:** NumPy, Pandas, Scikit-learn
* **GUI:** Tkinter

## 🧠 Model Architectures

### 1. Action Recognition Model
* **Backbone:** MobileNetV2 (Pre-trained on ImageNet).
* **Temporal Layer:** LSTM (128 units) to process sequences of 16 frames.
* **Input:** `(16 frames, 112, 112, 3)`.
* **Dataset:** UCF101 (Subset/Split).
* **Validation Accuracy:** ~85%.

### 2. Emotion Recognition Model
* **Backbone:** ResNet50V2 (Pre-trained on ImageNet).
* **Adaptation:** Custom `Conv2D` input layer to convert Grayscale (1-channel) input to RGB (3-channel) dynamically.
* **Input:** `(112, 112, 1)`.
* **Dataset:** FER-2013 (7 Classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise).
* **Validation Accuracy:** ~62.6% (Weighted Class Balancing applied).

## 📊 Performance & Benchmarks
Benchmarks recorded on test hardware (derived from `Evaluation.ipynb`):

| Model | Latency (ms) | FPS (approx) | Accuracy |
| :--- | :--- | :--- | :--- |
| **Action (MobileNetV2 + LSTM)** | ~87.36 ms | ~11.45 | 85.0% |
| **Emotion (ResNet50V2)** | ~91.00 ms | ~10.90 | 62.6% |

## 📂 Project Structure
```bash
.
├── Actions_Modelv2.ipynb   # Training script for the Action Recognition model
├── Emotions_Modelv2.ipynb  # Training script for the Emotion Recognition model
├── App.ipynb               # Main application (Inference & UI)
├── Evaluation.ipynb        # Latency and FPS benchmarking script
└── README.md
```
## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/Youssef-ashraff/Human-actions-and-facial-emotions.git](https://github.com/Youssef-ashraff/Human-actions-and-facial-emotions.git)
    cd Human-actions-and-facial-emotions
    ```

2.  **Install Dependencies**
    ```bash
    pip install tensorflow opencv-python numpy matplotlib scikit-learn tqdm
    ```

## 🖥️ Usage

1.  **Prepare Weights:** Ensure you have the trained model files (`best_sgd_model.h5` and `Emotions_best2.h5`) in the root directory.
2.  **Run the App:** You can run the application notebook directly or convert it to a script.
    ```bash
    # Option A: Run via Jupyter
    jupyter notebook App.ipynb
    
    # Option B: Run as script (if converted)
    python App.py
    ```
3.  **Select Input:** A menu will appear:
    * Select **1** for Webcam.
    * Select **2** to browse for a video file.
4.  **Controls:** Press **'q'** to quit the application window.


