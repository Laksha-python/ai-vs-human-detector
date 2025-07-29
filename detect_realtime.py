import cv2
import numpy as np
import torch
from torchvision import models, transforms
import mss
from PIL import Image
from collections import deque


monitor = {
    "top": 200,     
    "left": 600,    
    "width": 640,    
    "height": 1000   
}


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("fine_tuned_ai_vs_human.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


label_history = deque(maxlen=5)

def stable_label(new_label):
    label_history.append(new_label)
    return max(set(label_history), key=label_history.count)


def is_valid_face(face_crop):
    h, w, _ = face_crop.shape
    if h == 0 or w == 0:
        return False
    aspect_ratio = w / h
    if aspect_ratio < 0.5 or aspect_ratio > 1.5:
        return False
    brightness = np.mean(cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY))
    if brightness < 40: 
        return False
    return True


def predict_face(face_img):
    try:
        if not is_valid_face(face_img):
            return "Unknown"
        input_tensor = transform(face_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
            label_idx = np.argmax(probs)
            label = "AI" if label_idx == 1 else "Human"
            confidence = probs[label_idx]
            return f"{label} ({confidence:.2f})"
    except Exception as e:
        print("Prediction error:", e)
        return "Unknown"

print("🎥 Starting screen face detection... Press 'q' to quit.")

with mss.mss() as sct:
    while True:
        screenshot = sct.grab(monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            if w < 60 or h < 60:
                continue  

            margin = 10
            x1 = max(x - margin, 0)
            y1 = max(y - margin, 0)
            x2 = min(x + w + margin, frame.shape[1])
            y2 = min(y + h + margin, frame.shape[0])

            face_crop = frame[y1:y2, x1:x2]

            if not is_valid_face(face_crop):
                continue  

            raw_label = predict_face(face_crop)
            if raw_label == "Unknown":
                continue

            smoothed = stable_label(raw_label.split()[0])  
            confidence = raw_label.split()[1] if len(raw_label.split()) > 1 else ""
            label_text = f"{smoothed} {confidence}"

            color = (0, 255, 0) if smoothed == "Human" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("AI vs Human - Screen Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
