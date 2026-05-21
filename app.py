from flask import Flask, render_template, request
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import os
import pyttsx3
import threading
from werkzeug.utils import secure_filename

# Flask App
app = Flask(__name__)

# Upload Folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Model
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(
    torch.load("flower_model.pth", map_location=torch.device('cpu'))
)
model.eval()

# Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Labels
class_names = ['bud', 'post_receptive', 'receptive']


# ✅ FIX: Run speech in a separate thread to avoid pyttsx3 + Flask conflict
def speak(message):
    def run():
        try:
            engine = pyttsx3.init()        # ✅ Create a fresh engine each time
            engine.setProperty('rate', 165)
            engine.say(message)
            engine.runAndWait()
            engine.stop()                  # ✅ Clean up after speaking
        except Exception as e:
            print(f"Voice error: {e}")
    
    thread = threading.Thread(target=run)
    thread.daemon = True                   # ✅ Won't block Flask from shutting down
    thread.start()


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    image_path = None
    recommendation = None
    voice_message = None

    if request.method == "POST":

        file = request.files["image"]

        if file:

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print("IMAGE SAVED TO:", filepath)
            print("FILE EXISTS:", os.path.exists(filepath))  # debug check

            img = Image.open(filepath).convert("RGB")
            img_tensor = transform(img).unsqueeze(0)

            with torch.no_grad():
                outputs = model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence = probabilities[predicted.item()].item() * 100

            prediction = class_names[predicted.item()]
            image_path = filename

            if prediction == "bud":
                recommendation = (
                    "🌱 Flower is still in bud stage. "
                    "Pollination may be possible in 2–3 days."
                )
                voice_message = (
                    f"This flower is in bud stage "
                    f"with confidence {confidence:.2f} percent."
                )

            elif prediction == "receptive":
                recommendation = (
                    "🌸 Flower is in receptive stage. "
                    "Ready for pollination now."
                )
                voice_message = (
                    f"This flower is in receptive stage "
                    f"with confidence {confidence:.2f} percent."
                )

            else:
                recommendation = (
                    "🍂 Flower is in post receptive stage."
                )
                voice_message = (
                    f"This flower is in post receptive stage "
                    f"with confidence {confidence:.2f} percent."
                )

            # ✅ Speak in background — won't block or crash Flask
            speak(voice_message)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        recommendation=recommendation
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False    # ✅ Prevents double-init issues with pyttsx3
    )