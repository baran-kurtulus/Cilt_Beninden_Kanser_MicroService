import torch
import cv2
import numpy as np
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from pathlib import Path
import argparse


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self._conv_block(1, 16)
        self.enc2 = self._conv_block(16, 32)
        self.enc3 = self._conv_block(32, 64)
        self.enc4 = self._conv_block(64, 128)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self._conv_block(128, 256)

        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(32, 16)

        self.final = nn.Conv2d(16, 1, kernel_size=1)

    @staticmethod
    def _conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.final(d1)

def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="UNet skin segmentation test")
    parser.add_argument(
        "--weights",
        type=Path,
        default=script_dir / "Shifaa-Skin-Cancer-UNet-Segmentation.pth",
        help="Model weights file (.pth)"
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=script_dir / "ISICMalign1.jpg",
        help="Input image for segmentation"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "mask_output.png",
        help="Output mask file"
    )
    parser.add_argument(
        "--triple-output",
        type=Path,
        default=script_dir / "triple_output.png",
        help="3-panel output image (original + mask + overlay)"
    )
    parser.add_argument(
        "--cls-weights",
        type=Path,
        default=script_dir / "best_resnet50_skin_binary_final.pth",
        help="Classifier weights file (.pth), ResNet50 state_dict"
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default="benign,malignant",
        help="Comma-separated classifier labels in model output order"
    )
    parser.add_argument(
        "--no-classification",
        action="store_true",
        help="Skip classifier stage"
    )
    parser.add_argument(
        "--malignant-threshold",
        type=float,
        default=0.35,
        help="Decision threshold for malignant probability"
    )
    parser.add_argument(
        "--suspicious-margin",
        type=float,
        default=0.08,
        help="Uncertainty band around threshold (+/- margin)"
    )
    parser.add_argument(
        "--roi-min-area-ratio",
        type=float,
        default=0.01,
        help="Minimum mask area ratio to trust ROI classification"
    )
    parser.add_argument(
        "--roi-max-area-ratio",
        type=float,
        default=0.35,
        help="Maximum mask area ratio to trust ROI classification"
    )
    return parser.parse_args()


def make_triple_panel(original_bgr, binary_mask):
    h, w = original_bgr.shape[:2]
    mask_u8 = (binary_mask * 255).astype(np.uint8)
    mask_3ch = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)

    overlay = original_bgr.copy()
    red = np.zeros_like(original_bgr, dtype=np.uint8)
    red[:, :, 2] = 255
    alpha = 0.4
    lesion = binary_mask.astype(bool)
    overlay[lesion] = cv2.addWeighted(original_bgr, 1 - alpha, red, alpha, 0)[lesion]

    gap = 12
    label_h = 48
    panel_w = (w * 3) + (gap * 2)
    canvas = np.full((h + label_h, panel_w, 3), 255, dtype=np.uint8)

    x0 = 0
    x1 = w + gap
    x2 = (w * 2) + (gap * 2)
    canvas[:h, x0:x0 + w] = original_bgr
    canvas[:h, x1:x1 + w] = mask_3ch
    canvas[:h, x2:x2 + w] = overlay

    cv2.putText(canvas, "Original", (x0 + 10, h + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Mask", (x1 + 10, h + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Overlay", (x2 + 10, h + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
    return canvas


def largest_component_mask(mask):
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask, dtype=np.uint8)
    out[labels == largest_idx] = 1
    return out


def refine_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    cleaned = largest_component_mask(cleaned)
    return cleaned


def build_classifier(num_classes, device):
    clf = models.resnet50(weights=None)
    clf.fc = torch.nn.Linear(clf.fc.in_features, num_classes)
    return clf.to(device)


def crop_with_mask(image_bgr, mask, margin_ratio=0.12):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return image_bgr

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    h, w = image_bgr.shape[:2]

    mx = int((x1 - x0 + 1) * margin_ratio)
    my = int((y1 - y0 + 1) * margin_ratio)
    x0 = max(0, x0 - mx)
    y0 = max(0, y0 - my)
    x1 = min(w - 1, x1 + mx)
    y1 = min(h - 1, y1 + my)
    return image_bgr[y0:y1 + 1, x0:x1 + 1]


def run_classification(image_bgr, mask, cls_weights, class_names, device):
    if not cls_weights.exists():
        raise FileNotFoundError(f"Classifier weights file not found: {cls_weights}")

    classifier = build_classifier(num_classes=len(class_names), device=device)
    cls_state = torch.load(cls_weights, map_location=device)
    classifier.load_state_dict(cls_state)
    classifier.eval()

    roi_bgr = crop_with_mask(image_bgr, mask)
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

    cls_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    x_cls = cls_transform(roi_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = classifier(x_cls)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx = int(np.argmax(probs))
    return pred_idx, probs


def classify_patch(classifier, patch_bgr, device):
    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    cls_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    x_cls = cls_transform(patch_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = classifier(x_cls)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs


def get_malignant_index(class_names):
    lowered = [c.strip().lower() for c in class_names]
    for i, name in enumerate(lowered):
        if "mal" in name or "kotu" in name:
            return i
    if len(class_names) == 2:
        return 1
    return 0


def run_dual_classification(
    image_bgr,
    mask,
    cls_weights,
    class_names,
    device,
    roi_min_area_ratio,
    roi_max_area_ratio,
):
    if not cls_weights.exists():
        raise FileNotFoundError(f"Classifier weights file not found: {cls_weights}")

    classifier = build_classifier(num_classes=len(class_names), device=device)
    cls_state = torch.load(cls_weights, map_location=device)
    classifier.load_state_dict(cls_state)
    classifier.eval()

    full_probs = classify_patch(classifier, image_bgr, device)
    roi_patch = crop_with_mask(image_bgr, mask)
    roi_probs = classify_patch(classifier, roi_patch, device)

    area_ratio = float(mask.sum()) / float(mask.shape[0] * mask.shape[1])
    roi_trusted = roi_min_area_ratio <= area_ratio <= roi_max_area_ratio
    if roi_trusted:
        combined_probs = (full_probs + roi_probs) / 2.0
    else:
        combined_probs = full_probs
    pred_idx = int(np.argmax(combined_probs))
    return pred_idx, combined_probs, full_probs, roi_probs, area_ratio, roi_trusted


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights file not found: {args.weights}")
    if not args.image.exists():
        raise FileNotFoundError(f"Image file not found: {args.image}")

    model = UNet().to(device)
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # cv2.imread can fail on Windows paths with non-ASCII chars; imdecode is safer.
    image_bytes = np.fromfile(str(args.image), dtype=np.uint8)
    img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Image could not be read: {args.image}")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    x = transform(img_gray).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x)

    mask = torch.sigmoid(pred).squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    mask = refine_mask(mask)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), mask * 255)

    triple = make_triple_panel(img, mask)
    args.triple_output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.triple_output), triple)
    print(f"Mask saved to: {args.output}")
    print(f"3-panel saved to: {args.triple_output}")

    if args.no_classification:
        print("Classification skipped (--no-classification).")
        return

    class_names = [name.strip() for name in args.class_names.split(",") if name.strip()]
    if len(class_names) == 0:
        raise ValueError("At least one class name is required.")

    pred_idx, probs, probs_full, probs_roi, area_ratio, roi_trusted = run_dual_classification(
        image_bgr=img,
        mask=mask,
        cls_weights=args.cls_weights,
        class_names=class_names,
        device=device,
        roi_min_area_ratio=args.roi_min_area_ratio,
        roi_max_area_ratio=args.roi_max_area_ratio,
    )

    malignant_idx = get_malignant_index(class_names)
    p_malignant = float(probs[malignant_idx])
    low = max(0.0, args.malignant_threshold - args.suspicious_margin)
    high = min(1.0, args.malignant_threshold + args.suspicious_margin)
    if p_malignant >= high:
        decision = "malignant"
    elif p_malignant <= low:
        decision = "benign"
    else:
        decision = "suspicious"

    print("\nClassification:")
    print(f"Predicted class (argmax): {class_names[pred_idx]}")
    print(f"Final decision (thresholded): {decision}")
    print(f"Mask area ratio: {area_ratio:.4f} | ROI trusted: {roi_trusted}")
    print("Full-image probabilities:")
    for i, p in enumerate(probs_full):
        label = class_names[i] if i < len(class_names) else f"class_{i}"
        print(f"  {label}: {p:.4f}")
    print("ROI probabilities:")
    for i, p in enumerate(probs_roi):
        label = class_names[i] if i < len(class_names) else f"class_{i}"
        print(f"  {label}: {p:.4f}")
    print("Combined probabilities:")
    for i, p in enumerate(probs):
        label = class_names[i] if i < len(class_names) else f"class_{i}"
        print(f"  {label}: {p:.4f}")
    print(
        f"malignant_probability={p_malignant:.4f}, "
        f"threshold={args.malignant_threshold:.2f}, "
        f"suspicious_band=[{low:.2f}, {high:.2f}]"
    )


if __name__ == "__main__":
    main()
