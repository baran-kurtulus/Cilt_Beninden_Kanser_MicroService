import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
import torchvision.models as models


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="ResNet50 skin lesion binary classification"
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=script_dir / "ISICMalign1.jpg",
        help="Input image for classification",
    )
    parser.add_argument(
        "--cls-weights",
        type=Path,
        default=script_dir / "resnet50_ham10000_isic2019_binary_best_epoch19.pth",
        help="Classifier weights file (.pth), ResNet50 state_dict",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default="benign,malignant",
        help="Comma-separated classifier labels in model output order",
    )
    parser.add_argument(
        "--malignant-threshold",
        type=float,
        default=0.35,
        help="Decision threshold for malignant probability",
    )
    parser.add_argument(
        "--suspicious-margin",
        type=float,
        default=0.08,
        help="Uncertainty band around threshold (+/- margin)",
    )
    return parser.parse_args()


def build_classifier(num_classes, device):
    classifier = models.resnet50(weights=None)
    classifier.fc = torch.nn.Linear(classifier.fc.in_features, num_classes)
    return classifier.to(device)


def load_state_dict(weights_path, device):
    try:
        return torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(weights_path, map_location=device)


def read_image(image_path):
    image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Image could not be read: {image_path}")
    return image


def classify_image(classifier, image_bgr, device):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    input_tensor = transform(image_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = classifier(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probabilities


def get_malignant_index(class_names):
    lowered = [name.strip().lower() for name in class_names]
    for index, name in enumerate(lowered):
        if "mal" in name or "kotu" in name:
            return index
    if len(class_names) == 2:
        return 1
    return 0


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.image.exists():
        raise FileNotFoundError(f"Image file not found: {args.image}")
    if not args.cls_weights.exists():
        raise FileNotFoundError(
            f"Classifier weights file not found: {args.cls_weights}"
        )

    class_names = [
        name.strip() for name in args.class_names.split(",") if name.strip()
    ]
    if not class_names:
        raise ValueError("At least one class name is required.")

    classifier = build_classifier(len(class_names), device)
    state_dict = load_state_dict(args.cls_weights, device)
    classifier.load_state_dict(state_dict)
    classifier.eval()

    image = read_image(args.image)
    probabilities = classify_image(classifier, image, device)
    predicted_index = int(np.argmax(probabilities))

    malignant_index = get_malignant_index(class_names)
    malignant_probability = float(probabilities[malignant_index])
    low = max(0.0, args.malignant_threshold - args.suspicious_margin)
    high = min(1.0, args.malignant_threshold + args.suspicious_margin)

    if malignant_probability >= high:
        decision = "malignant"
    elif malignant_probability <= low:
        decision = "benign"
    else:
        decision = "suspicious"

    print("\nClassification:")
    print(f"Predicted class (argmax): {class_names[predicted_index]}")
    print(f"Final decision (thresholded): {decision}")
    print("Full-image probabilities:")
    for index, probability in enumerate(probabilities):
        label = (
            class_names[index]
            if index < len(class_names)
            else f"class_{index}"
        )
        print(f"  {label}: {probability:.4f}")
    print(
        f"malignant_probability={malignant_probability:.4f}, "
        f"threshold={args.malignant_threshold:.2f}, "
        f"suspicious_band=[{low:.2f}, {high:.2f}]"
    )


if __name__ == "__main__":
    main()
