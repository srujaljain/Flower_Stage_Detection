import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import pyttsx3

# Initialize voice engine
engine = pyttsx3.init()
plant_name = "Papaya"

# Load model
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 3)

# Load trained model
model.load_state_dict(torch.load("flower_model.pth"))

# Set evaluation mode
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load image
img = Image.open("test(12).jpg")
img = transform(img).unsqueeze(0)

# Class names (same order used during training)
class_names = ['bud', 'post_receptive', 'receptive']

# Prediction
with torch.no_grad():

    # Model output
    outputs = model(img)

    # Predicted class index
    _, predicted = torch.max(outputs, 1)

    # Confidence calculation
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    confidence = probabilities[predicted.item()].item() * 100

# Predicted label
label = class_names[predicted.item()]

# Display output
print("🔍 Predicted class:", label)
print(f"📊 Confidence: {confidence:.2f}%")
print("🌿 Plant Name:", plant_name)

# Voice + Recommendation Messages
if label == "bud":

    message = (
        "This flower is still in bud stage. "
        "It may be ready for pollination in 2 to 3 days."
    )

    print("🌱 This flower is still in bud stage.")
    print("⏳ It may be ready for pollination in 2–3 days.")
    print("👉 Recommended Action: Monitor flower growth daily.")

    engine.say(message)
    engine.runAndWait()

elif label == "receptive":

    message = (
        "This flower is in receptive stage. "
        "It is ready for pollination now."
    )

    print("🌸 This flower is in receptive stage.")
    print("✅ It is ready for pollination now!")
    print("👉 Recommended Action: Start pollination between 6 AM and 10 AM.")

    engine.say(message)
    engine.runAndWait()

elif label == "post_receptive":

    message = (
        "This flower is in post receptive stage. "
        "Pollination period has passed."
    )

    print("🍂 This flower is in post-receptive stage.")
    print("❌ Pollination period has passed.")
    print("⚠️ Recommended Action: Monitor fruit development.")

    engine.say(message)
    engine.runAndWait()