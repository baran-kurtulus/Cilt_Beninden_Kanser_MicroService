import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
UNET_WEIGHTS = BASE_DIR / "Shifaa-Skin-Cancer-UNet-Segmentation.pth"
RESNET_WEIGHTS = BASE_DIR / "best_resnet50_skin_binary_final.pth"
MODEL_VERSION = "resnet50-v1.0"
CLASS_NAMES = ["benign", "malignant"]
UNET_IMAGE_SIZE = 256
CLS_IMAGE_SIZE = 224
ROI_MIN_AREA_RATIO = 0.01
ROI_MAX_AREA_RATIO = 0.35


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


def largest_component_mask(mask: np.ndarray) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask, dtype=np.uint8)
    out[labels == largest_idx] = 1
    return out


def refine_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    cleaned = largest_component_mask(cleaned)
    return cleaned


def build_classifier(num_classes: int, device: torch.device) -> nn.Module:
    clf = models.resnet50(weights=None)
    clf.fc = nn.Linear(clf.fc.in_features, num_classes)
    return clf.to(device)


def crop_with_mask(
    image_bgr: np.ndarray, mask: np.ndarray, margin_ratio: float = 0.12
) -> np.ndarray:
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
    return image_bgr[y0 : y1 + 1, x0 : x1 + 1]


_cls_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((CLS_IMAGE_SIZE, CLS_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


def classify_patch(
    classifier: nn.Module, patch_bgr: np.ndarray, device: torch.device
) -> np.ndarray:
    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    x = _cls_transform(patch_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = classifier(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs


def get_malignant_index(class_names: list) -> int:
    lowered = [c.strip().lower() for c in class_names]
    for i, name in enumerate(lowered):
        if "mal" in name or "kotu" in name:
            return i
    if len(class_names) == 2:
        return 1
    return 0


def run_segmentation(
    unet: nn.Module, bgr_image: np.ndarray, device: torch.device
) -> np.ndarray:
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    unet_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((UNET_IMAGE_SIZE, UNET_IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )
    x = unet_transform(gray).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = unet(x)
    mask = torch.sigmoid(pred).squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8)
    mask = cv2.resize(mask, (bgr_image.shape[1], bgr_image.shape[0]))
    mask = refine_mask(mask)
    return mask


def run_dual_classification(
    classifier: nn.Module,
    bgr_image: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    roi_min: float = ROI_MIN_AREA_RATIO,
    roi_max: float = ROI_MAX_AREA_RATIO,
) -> dict:
    full_probs = classify_patch(classifier, bgr_image, device)
    roi_patch = crop_with_mask(bgr_image, mask)
    roi_probs = classify_patch(classifier, roi_patch, device)
    area_ratio = float(mask.sum()) / float(mask.shape[0] * mask.shape[1])
    roi_trusted = roi_min <= area_ratio <= roi_max
    if roi_trusted:
        combined_probs = (full_probs + roi_probs) / 2.0
    else:
        combined_probs = full_probs
    pred_idx = int(np.argmax(combined_probs))
    return {
        "pred_idx": pred_idx,
        "combined_probs": combined_probs,
        "full_probs": full_probs,
        "roi_probs": roi_probs,
        "area_ratio": area_ratio,
        "roi_trusted": roi_trusted,
    }


# ---------------------------------------------------------------------------
# Global model references (set during lifespan startup)
# ---------------------------------------------------------------------------
unet_model: Optional[nn.Module] = None
classifier_model: Optional[nn.Module] = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load both AI models on startup."""
    global unet_model, classifier_model

    logger.info(f"Using device: {DEVICE}")

    if not UNET_WEIGHTS.exists():
        raise FileNotFoundError(f"UNet weights not found: {UNET_WEIGHTS}")
    if not RESNET_WEIGHTS.exists():
        raise FileNotFoundError(f"ResNet weights not found: {RESNET_WEIGHTS}")

    # Load UNet
    unet_model = UNet().to(DEVICE)
    unet_state = torch.load(UNET_WEIGHTS, map_location=DEVICE)
    unet_model.load_state_dict(unet_state)
    unet_model.eval()
    logger.info("UNet segmentation model loaded.")

    # Load ResNet50 classifier
    classifier_model = build_classifier(len(CLASS_NAMES), DEVICE)
    cls_state = torch.load(RESNET_WEIGHTS, map_location=DEVICE)
    classifier_model.load_state_dict(cls_state)
    classifier_model.eval()
    logger.info("ResNet50 classification model loaded.")

    yield

    # Cleanup
    unet_model = None
    classifier_model = None
    logger.info("Models unloaded.")


app = FastAPI(
    title="Skin Cancer Analysis Microservice",
    description="UNet + ResNet50 skin lesion segmentation and classification",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "application/octet-stream"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "device": str(DEVICE),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if unet_model is None or classifier_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    content_type = file.content_type or "application/octet-stream"
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {content_type}. Allowed: JPEG, PNG.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    logger.info(
        "Predicting for %s | %dx%d | %.1f KB",
        file.filename,
        bgr.shape[1],
        bgr.shape[0],
        len(image_bytes) / 1024,
    )

    mask = run_segmentation(unet_model, bgr, DEVICE)

    result = run_dual_classification(
        classifier_model,
        bgr,
        mask,
        DEVICE,
    )

    combined = result["combined_probs"]
    malignant_idx = get_malignant_index(CLASS_NAMES)
    p_malignant = float(combined[malignant_idx])

    if p_malignant > 0.5:
        label = "malignant"
        confidence = p_malignant
    else:
        label = "benign"
        confidence = 1.0 - p_malignant

    confidence = round(confidence, 4)

    logger.info(
        "Result: label=%s confidence=%.4f area_ratio=%.4f roi_trusted=%s",
        label,
        confidence,
        result["area_ratio"],
        result["roi_trusted"],
    )

    return {
        "label": label,
        "confidence": confidence,
        "model_version": MODEL_VERSION,
    }
