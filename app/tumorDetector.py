import io
import os
import sys

import numpy as np

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

import streamlit as st

from matplotlib import cm 
from scipy.ndimage import gaussian_filter

from PIL import Image, ImageEnhance
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import TransferResNetLarge
from src.dataloader import IMAGENET_MEAN, IMAGENET_STD, LABEL_NAMES

# ----------------------
# Config
# ----------------------
MODEL_WEIGHTS_PATH = Path(ROOT / "src/final_models/transferResNetLarge.th").resolve()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------
# Grad-CAM Function
# ----------------------
def generate_gradcam(
    model,
    image_path,
    model_name="TransferResNetLarge",
    class_names=None,
    device=None,
    colormap="inferno",
    apply_smoothing=True,
    brightness_factor=1.0,
    return_images=False,
    show=False
):
    """
    Grad-CAM function over TransferResNetLarge
    Returns overlay and base images when return_iamges=True.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # target layer for TransferResNetLarge
    target_layer = model.base_model.layer4
    mode = "pretrained"

    activations, gradients = [], []

    def forward_hook(_, __, output):
        activations.append(output)

    def backward_hook(_, __, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    # load raw and transformed image
    image = Image.open(image_path).convert("RGB")
    transform_raw = Compose([Resize((224, 224)), ToTensor()])
    image_tensor_raw = transform_raw(image)

    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    # forward and backward pass
    output = model(input_tensor)
    pred_class = output.argmax().item()
    class_label = class_names[pred_class] if class_names else f"class {pred_class}"
    model.zero_grad()
    output[0, pred_class].backward()

    # GradCAM calc
    act = activations[0].squeeze(0)
    grad = gradients[0].squeeze(0)
    weights = grad.mean(dim=(1, 2))
    cam = F.relu((weights[:, None, None] * act).sum(0))
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)
    cam_np = cam.detach().cpu().numpy()
    if apply_smoothing:
        cam_np = gaussian_filter(cam_np, sigma=0.5)

    # heatmap image
    cmap_ = cm.get_cmap(colormap)
    target_size = (224, 224)
    cam_img = Image.fromarray(np.uint8(cmap_(cam_np)[:, :, :3] * 255)).resize(target_size)

    # denormalize the raw image
    mean = np.array(IMAGENET_MEAN)[:, None, None]
    std = np.array(IMAGENET_STD)[:, None, None]
    img_raw = image_tensor_raw.squeeze(0).detach().cpu().numpy()
    img_raw = np.clip(img_raw * std + mean, 0, 1).transpose(1, 2, 0)
    base_img = Image.fromarray(np.uint8(img_raw * 255)).convert("RGB")

    # apply brightness adjustment
    if brightness_factor != 1.0:
        base_img = ImageEnhance.Brightness(base_img).enhance(brightness_factor)

    # overlay CAM on base image
    base_img_rgba = base_img.convert("RGBA")
    cam_overlay = Image.blend(base_img_rgba, cam_img.convert("RGBA"), alpha=0.7)

    # clean up hooks
    forward_handle.remove()
    backward_handle.remove()
    activations.clear()
    gradients.clear()

    if return_images:
        return {
            "base_img": base_img,
            "overlay": cam_overlay.convert("RGB"),
            "pred_idx": pred_class,
            "pred_label": class_label
        }
    
# ----------------------
# Model Loading
# ----------------------
@st.cache_resource(show_spinner=False)
def load_model():
    model = TransferResNetLarge(num_classes=4)
    assert MODEL_WEIGHTS_PATH.exists(), f"Weights not found at {MODEL_WEIGHTS_PATH}"
    state = torch.load(MODEL_WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model.to(DEVICE)

# ----------------------
# Inference Utility
# ----------------------
@torch.no_grad()
def predict(model, pil_image):
    preprocess = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    x = preprocess(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)
    logits = model(x)
    pred_idx = int(torch.argmax(logits, dim=1).item())
    pred_label = LABEL_NAMES[pred_idx]
    return pred_idx, pred_label

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="Brain MRI Tumor Detector", layout="centered")
st.title("Brain MRI Tumor Detector")
st.caption("Upload a single image. The app flags if a tumor is detected and shows possible tumor location.")

model = load_model()

uploaded = st.file_uploader("Upload an MRI image (JPG/PNG)",
                            type=["jpg", "jpeg", "png"])

if uploaded is not None:
    # Read image
    pil_img = Image.open(uploaded).convert("RGB")
    st.image(pil_img, caption="Uploaded Image", use_container_width=True)

    # predict
    pred_idx, pred_label = predict(model, pil_img)

    if pred_label == "notumor":
        st.success("No tumor detected.")
    else:
        st.error("Tumor detected.")
        # Run Grad-CAM
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)

        # Generate the Grad-CAM
        result = generate_gradcam(
            model=model,
            image_path=buf,
            class_names=LABEL_NAMES,
            device=DEVICE,
            return_images=True,
            show=False
        )

        # show images
        col1, col2 = st.columns(2)
        with col1:
            st.image(result["base_img"], 
                     caption="Original Image",
                     use_container_width=True)
        with col2:
            st.image(result["overlay"],
                     caption="Grad-CAM Overlay",
                     use_container_width=True)