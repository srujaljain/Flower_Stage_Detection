from flask import Flask, render_template, request
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import os

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model setup
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 3)

# Load weights
model.load_state_dict(
    torch.load(
        "flower_model.pth",
        map_location=torch.device('cpu')
    )
)

model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Labels
class_names = ['bud', 'post_receptive', 'receptive']

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":

        file = request.files["image"]

        if file:

            filepath = os.path.join(
                UPLOAD_FOLDER,
                file.filename
            )

            file.save(filepath)

            # Open image
            img = Image.open(filepath).convert("RGB")

            # Transform
            img = transform(img).unsqueeze(0)

            # Predict
            with torch.no_grad():

                outputs = model(img)

                _, predicted = torch.max(outputs, 1)

                probabilities = torch.nn.functional.softmax(
                    outputs[0],
                    dim=0
                )

                confidence = (
                    probabilities[predicted.item()].item()
                    * 100
                )

            prediction = class_names[predicted.item()]

            image_path = filepath

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )

if __name__ == "__main__":
    app.run(debug=True)
    