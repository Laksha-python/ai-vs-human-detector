# 🤖 AI vs Human Face Classification  
Real-Time Face Detection + ResNet18 Model (PyTorch)

This repository contains two main components:

1. **Model Training Script** – `train_finetune.py`  
   Fine-tunes a ResNet18 model to classify faces as **AI-generated** or **Human**.

2. **Real-Time Detection Script** – `detect_realtime.py`  
   Uses webcam/screen capture + Haarcascade + PyTorch model to detect faces and classify them live.

---

## 🔥 Features

### ✅ Model Training (`train_finetune.py`)
- Downloads dataset from Google Drive  
- Fine-tunes **ResNet18**  
- Early image transforms + dataloaders  
- Saves trained model as `fine_tuned_ai_vs_human.pth`

---

### 🎥 Real-Time Detection (`detect_realtime.py`)
- Captures screen region using **MSS**
- Detects faces using **Haarcascade**
- Classifies each face as **AI** or **Human**
- Confidence display  
- Label smoothing using Deque  
- Real-time bounding boxes + color coding  
  - 🟢 Human  
  - 🔴 AI

---

## 📁 Project Structure
ai-vs-human-detector/
│── detect_realtime.py
│── train_finetune.py
│── fine_tuned_ai_vs_human.pth
│── requirements.txt
│── README.md
│
└── dataset/
├── train/
├── validate/
└── test/
---

## 🚀 Installation

### 1️⃣ Clone the repo

git clone https://github.com/yourusername/ai-vs-human-detector.git

cd ai-vs-human-detector

2️⃣ Install dependencies

pip install -r requirements.txt

🧠 Model Training

Run training script:

python train_finetune.py

Expected Output:

Dataset is downloaded & extracted

ResNet18 is fine-tuned for 5 epochs

Final model saved at:
fine_tuned_ai_vs_human.pth

Replace the dataset Google Drive file ID here:

DATASET_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"

🎥 Real-Time Face Classification

Make sure fine_tuned_ai_vs_human.pth is in the same folder as detect_realtime.py.

Run detection:

python detect_realtime.py

Controls:

Press q → Quit

Real-time window:

Green box = Human

Red box = AI

Smoothed labels for accuracy

⚙️ How It Works

🟩 1. Face Detection

Uses OpenCV Haarcascade to detect faces on screen:

face_cascade = cv2.CascadeClassifier(...)

🟦 2. Face Validation

Removes bad, dark, cropped, invalid faces:

is_valid_face(face_crop)

🟥 3. ResNet18 Inference

Transforms → Tensor → Feed to model:

output = model(input_tensor)

🟨 4. Stable Labeling

Smooths predictions over last 5 frames:

stable_label()

📦 Requirements

torch

torchvision

opencv-python

numpy

mss

pillow

tqdm

gdown

Install all:

pip install -r requirements.txt

🛠 Future Improvements

Add GUI (Streamlit or Flask)

Support webcam detection

Add dataset augmentation

Export model to ONNX for mobile use

📜 License

MIT License – Free to use and modify.

✨ Author

Laksha Karimuthu



