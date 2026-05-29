import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms

# -----------------------
# PATHS
# -----------------------
model_path = r"C:\Users\CumFur\Desktop\Furkan\VS_Code\Bitirme\8epoch_skin_model.pth"
image_path = r"C:\Users\CumFur\Desktop\Furkan\VS_Code\Bitirme\test2.jpg"

# -----------------------
# AYARLAR
# -----------------------
num_classes = 2
class_names = ["class_0", "class_1"]  # kendi sınıflarınla değiştir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# MODEL (RESNET50)
# -----------------------
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# state_dict yükle
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)

model = model.to(device)
model.eval()

# -----------------------
# IMAGE PREPROCESS
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# -----------------------
# INFERENCE
# -----------------------
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)
    pred = torch.argmax(probs, dim=1).item()

print("Tahmin index:", pred)
print("Tahmin sınıf:", class_names[pred])
print("Olasılıklar:", probs.cpu().numpy())