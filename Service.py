import base64
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile
from torchvision import transforms

from Saglam_Calisan_kod import (
    UNet,
    build_classifier,
    crop_with_mask,
    get_malignant_index,
    make_triple_panel,
    refine_mask,
)


BASE_DIR = Path(__file__).resolve().parent


def env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    path = Path(value).expanduser() if value else default
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


SEGMENTATION_WEIGHTS = env_path(
    "SEGMENTATION_WEIGHTS",
    BASE_DIR / "Shifaa-Skin-Cancer-UNet-Segmentation.pth",
)
CLASSIFIER_WEIGHTS = env_path(
    "CLASSIFIER_WEIGHTS",
    BASE_DIR / "best_resnet50_skin_binary_final.pth",
)
CLASS_NAMES = [
    name.strip()
    for name in os.getenv("CLASS_NAMES", "benign,malignant").split(",")
    if name.strip()
]
MODEL_VERSION = os.getenv("MODEL_VERSION", "unet-resnet50-v1")

DEFAULT_MALIGNANT_THRESHOLD = float(os.getenv("MALIGNANT_THRESHOLD", "0.35"))
DEFAULT_SUSPICIOUS_MARGIN = float(os.getenv("SUSPICIOUS_MARGIN", "0.08"))
DEFAULT_ROI_MIN_AREA_RATIO = float(os.getenv("ROI_MIN_AREA_RATIO", "0.01"))
DEFAULT_ROI_MAX_AREA_RATIO = float(os.getenv("ROI_MAX_AREA_RATIO", "0.35"))

DEVICE = torch.device(os.getenv("TORCH_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))

SEGMENT_TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)

CLASSIFIER_TRANSFORM = transforms.Compose(
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

segmentation_model: UNet | None = None
classifier_model: torch.nn.Module | None = None


def load_models() -> None:
    global segmentation_model, classifier_model

    if not CLASS_NAMES:
        raise RuntimeError("CLASS_NAMES must contain at least one label.")
    if not SEGMENTATION_WEIGHTS.exists():
        raise FileNotFoundError(f"Segmentation weights not found: {SEGMENTATION_WEIGHTS}")
    if not CLASSIFIER_WEIGHTS.exists():
        raise FileNotFoundError(f"Classifier weights not found: {CLASSIFIER_WEIGHTS}")

    if segmentation_model is None:
        model = UNet().to(DEVICE)
        state_dict = torch.load(SEGMENTATION_WEIGHTS, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        segmentation_model = model

    if classifier_model is None:
        model = build_classifier(num_classes=len(CLASS_NAMES), device=DEVICE)
        state_dict = torch.load(CLASSIFIER_WEIGHTS, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        classifier_model = model


def ensure_models_loaded() -> None:
    if segmentation_model is None or classifier_model is None:
        load_models()


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_models()
    yield


app = FastAPI(
    title="Skin Cancer Detection Microservice",
    description=(
        "Uploads a skin lesion image, creates a UNet mask, then runs the "
        "ResNet50 classifier on the full image and ROI."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def decode_image(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise ValueError("Image body is empty.")

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image could not be decoded. Send a valid JPG/PNG image.")
    return image


async def read_image_bytes(request: Request) -> tuple[bytes, str | None]:
    content_type = request.headers.get("content-type", "").split(";")[0].lower()

    if content_type == "application/json":
        payload = await request.json()
        encoded = payload.get("image_base64") or payload.get("image")
        if not isinstance(encoded, str):
            raise ValueError("JSON body must include an image_base64 field.")
        if encoded.startswith("data:") and "," in encoded:
            encoded = encoded.split(",", 1)[1]
        try:
            return base64.b64decode(encoded, validate=True), payload.get("filename")
        except ValueError as exc:
            raise ValueError("image_base64 is not valid base64.") from exc

    if content_type == "multipart/form-data":
        try:
            form = await request.form()
        except Exception as exc:
            raise ValueError(
                "Multipart upload requires python-multipart. "
                "Install it with: pip install python-multipart"
            ) from exc

        upload = form.get("file") or form.get("image")
        if upload is None:
            upload = next(
                (value for value in form.values() if isinstance(value, StarletteUploadFile)),
                None,
            )
        if not isinstance(upload, StarletteUploadFile):
            raise ValueError("Multipart body must contain a file field named file or image.")
        return await upload.read(), upload.filename

    return await request.body(), request.headers.get("x-filename")


def segment_image(image_bgr: np.ndarray) -> np.ndarray:
    ensure_models_loaded()
    assert segmentation_model is not None

    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    x = SEGMENT_TRANSFORM(image_gray).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = segmentation_model(x)

    mask = torch.sigmoid(pred).squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8)
    mask = cv2.resize(
        mask,
        (image_bgr.shape[1], image_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return refine_mask(mask)


def classify_patch_with_loaded_model(patch_bgr: np.ndarray) -> np.ndarray:
    ensure_models_loaded()
    assert classifier_model is not None

    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    x = CLASSIFIER_TRANSFORM(patch_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = classifier_model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs


def classify_image(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    roi_min_area_ratio: float,
    roi_max_area_ratio: float,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, float, bool]:
    full_probs = classify_patch_with_loaded_model(image_bgr)
    roi_patch = crop_with_mask(image_bgr, mask)
    roi_probs = classify_patch_with_loaded_model(roi_patch)

    area_ratio = float(mask.sum()) / float(mask.shape[0] * mask.shape[1])
    roi_trusted = roi_min_area_ratio <= area_ratio <= roi_max_area_ratio
    combined_probs = (full_probs + roi_probs) / 2.0 if roi_trusted else full_probs
    pred_idx = int(np.argmax(combined_probs))
    return pred_idx, combined_probs, full_probs, roi_probs, area_ratio, roi_trusted


def probabilities_to_dict(probs: np.ndarray) -> dict[str, float]:
    return {
        CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}": round(float(prob), 6)
        for i, prob in enumerate(probs)
    }


def make_decision(
    probs: np.ndarray,
    malignant_threshold: float,
    suspicious_margin: float,
) -> tuple[str, float, float, float]:
    malignant_idx = get_malignant_index(CLASS_NAMES)
    malignant_probability = float(probs[malignant_idx])
    low = max(0.0, malignant_threshold - suspicious_margin)
    high = min(1.0, malignant_threshold + suspicious_margin)

    if malignant_probability >= high:
        decision = "malignant"
    elif malignant_probability <= low:
        decision = "benign"
    else:
        decision = "suspicious"

    return decision, malignant_probability, low, high


def get_decision_confidence(decision: str, malignant_probability: float) -> float:
    if decision == "malignant":
        return malignant_probability
    if decision == "benign":
        return 1.0 - malignant_probability
    return max(malignant_probability, 1.0 - malignant_probability)


def encode_png_base64(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Image could not be encoded as PNG.")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def predict_image(
    image_bgr: np.ndarray,
    malignant_threshold: float,
    suspicious_margin: float,
    roi_min_area_ratio: float,
    roi_max_area_ratio: float,
    return_images: bool,
) -> dict[str, Any]:
    mask = segment_image(image_bgr)
    pred_idx, probs, full_probs, roi_probs, area_ratio, roi_trusted = classify_image(
        image_bgr=image_bgr,
        mask=mask,
        roi_min_area_ratio=roi_min_area_ratio,
        roi_max_area_ratio=roi_max_area_ratio,
    )
    decision, malignant_probability, low, high = make_decision(
        probs=probs,
        malignant_threshold=malignant_threshold,
        suspicious_margin=suspicious_margin,
    )
    confidence = get_decision_confidence(decision, malignant_probability)

    result: dict[str, Any] = {
        "label": decision,
        "confidence": round(confidence, 6),
        "model_version": MODEL_VERSION,
        "decision": decision,
        "predicted_class": CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"class_{pred_idx}",
        "predicted_index": pred_idx,
        "malignant_probability": round(malignant_probability, 6),
        "threshold": malignant_threshold,
        "suspicious_band": {"low": round(low, 6), "high": round(high, 6)},
        "mask": {
            "area_ratio": round(area_ratio, 6),
            "roi_trusted": roi_trusted,
        },
        "probabilities": {
            "combined": probabilities_to_dict(probs),
            "full_image": probabilities_to_dict(full_probs),
            "roi": probabilities_to_dict(roi_probs),
        },
    }

    if return_images:
        result["images"] = {
            "mask_png_base64": encode_png_base64(mask * 255),
            "triple_panel_png_base64": encode_png_base64(make_triple_panel(image_bgr, mask)),
        }

    return result


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "skin-cancer-detection",
        "status": "running",
        "predict_endpoint": "POST /predict",
        "accepted_bodies": [
            "raw image bytes with content-type image/jpeg, image/png, or application/octet-stream",
            "JSON with image_base64",
            "multipart/form-data with file field if python-multipart is installed",
        ],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "device": str(DEVICE),
        "segmentation_weights": SEGMENTATION_WEIGHTS.name,
        "classifier_weights": CLASSIFIER_WEIGHTS.name,
        "class_names": CLASS_NAMES,
        "models_loaded": segmentation_model is not None and classifier_model is not None,
    }


@app.post("/predict")
async def predict(
    request: Request,
    return_images: bool = Query(False, description="Return mask and preview images as base64 PNG."),
    malignant_threshold: float = Query(
        DEFAULT_MALIGNANT_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Malignant probability threshold.",
    ),
    suspicious_margin: float = Query(
        DEFAULT_SUSPICIOUS_MARGIN,
        ge=0.0,
        le=1.0,
        description="Uncertainty band around malignant_threshold.",
    ),
    roi_min_area_ratio: float = Query(
        DEFAULT_ROI_MIN_AREA_RATIO,
        ge=0.0,
        le=1.0,
        description="Minimum mask area ratio to trust ROI classification.",
    ),
    roi_max_area_ratio: float = Query(
        DEFAULT_ROI_MAX_AREA_RATIO,
        ge=0.0,
        le=1.0,
        description="Maximum mask area ratio to trust ROI classification.",
    ),
) -> dict[str, Any]:
    if roi_min_area_ratio > roi_max_area_ratio:
        raise HTTPException(status_code=400, detail="roi_min_area_ratio cannot exceed roi_max_area_ratio.")

    try:
        image_bytes, filename = await read_image_bytes(request)
        image_bgr = decode_image(image_bytes)
        result = predict_image(
            image_bgr=image_bgr,
            malignant_threshold=malignant_threshold,
            suspicious_margin=suspicious_margin,
            roi_min_area_ratio=roi_min_area_ratio,
            roi_max_area_ratio=roi_max_area_ratio,
            return_images=return_images,
        )
        if filename:
            result["filename"] = filename
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict-file")
async def predict_file(
    file: UploadFile = File(..., description="JPG or PNG skin lesion image."),
    return_images: bool = Query(False, description="Return mask and preview images as base64 PNG."),
    malignant_threshold: float = Query(
        DEFAULT_MALIGNANT_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Malignant probability threshold.",
    ),
    suspicious_margin: float = Query(
        DEFAULT_SUSPICIOUS_MARGIN,
        ge=0.0,
        le=1.0,
        description="Uncertainty band around malignant_threshold.",
    ),
    roi_min_area_ratio: float = Query(
        DEFAULT_ROI_MIN_AREA_RATIO,
        ge=0.0,
        le=1.0,
        description="Minimum mask area ratio to trust ROI classification.",
    ),
    roi_max_area_ratio: float = Query(
        DEFAULT_ROI_MAX_AREA_RATIO,
        ge=0.0,
        le=1.0,
        description="Maximum mask area ratio to trust ROI classification.",
    ),
) -> dict[str, Any]:
    if roi_min_area_ratio > roi_max_area_ratio:
        raise HTTPException(status_code=400, detail="roi_min_area_ratio cannot exceed roi_max_area_ratio.")

    try:
        image_bgr = decode_image(await file.read())
        result = predict_image(
            image_bgr=image_bgr,
            malignant_threshold=malignant_threshold,
            suspicious_margin=suspicious_margin,
            roi_min_area_ratio=roi_min_area_ratio,
            roi_max_area_ratio=roi_max_area_ratio,
            return_images=return_images,
        )
        result["filename"] = file.filename
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )
