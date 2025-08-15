import io
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk, ImageEnhance

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from matplotlib import cm
from scipy.ndimage import gaussian_filter

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import TransferResNetLarge
from src.dataloader import IMAGENET_MEAN, IMAGENET_STD, LABEL_NAMES

# ----------------------
# Config
# ----------------------
WEIGHTS_PATH = (ROOT / "src/final_models/transferResNetLarge.th").resolve()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------
# Grad-CAM (single model)
# ----------------------
def generate_gradcam(
    model,
    image_path,
    class_names=None,
    device=None,
    colormap='inferno',
    apply_smoothing=True,
    brightness_factor=1.0,
    return_images=False
):
    """
    Grad-CAM for TransferResNetLarge (hooks: base_model.layer4).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    target_layer = model.base_model.layer4

    activations, gradients = [], []

    def forward_hook(_, __, output):
        activations.append(output)

    def backward_hook(_, __, grad_output):
        gradients.append(grad_output[0])

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    # load image and set transforms (ImageNet stats)
    image = Image.open(image_path).convert("RGB")
    transform_raw = Compose([
        Resize((224, 224)),
        ToTensor()
    ])
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

    # Grad-CAM
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
    cam_img = Image.fromarray(np.uint8(cmap_(cam_np)[:, :, :3] * 255)).resize((224, 224))

    # denorm raw image
    mean = np.array(IMAGENET_MEAN)[:, None, None]
    std = np.array(IMAGENET_STD)[:, None, None]
    img_raw = image_tensor_raw.squeeze(0).detach().cpu().numpy()
    img_raw = np.clip(img_raw * std + mean, 0, 1).transpose(1, 2, 0)
    base_img = Image.fromarray(np.uint8(img_raw * 255)).convert("RGB")

    if brightness_factor != 1.0:
        base_img = ImageEnhance.Brightness(base_img).enhance(brightness_factor)

    cam_overlay = Image.blend(base_img.convert("RGBA"), cam_img.convert("RGBA"), alpha=0.7)

    fh.remove()
    bh.remove()
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
# Model loading & inference
# ----------------------
@torch.no_grad()
def load_model():
    model = TransferResNetLarge(num_classes=len(LABEL_NAMES))
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Weights not found at {WEIGHTS_PATH}")
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model.to(DEVICE)

@torch.no_grad()
def predict(model, pil_image: Image.Image):
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
# Tkinter App
# ----------------------
class TumorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Brain MRI Tumor Detector")
        self.geometry("980x640")
        self.resizable(True, True)

        # model
        try:
            self.model = load_model()
        except Exception as e:
            messagebox.showerror("Model Load Error", str(e))
            raise

        # state: keep th elast PIL images show
        self.left_pil: Image.Image | None = None
        self.right_pil: Image.Image | None = None

        # UI
        self._build_ui()

        # freeze panel sizes so children don't drive layout (prevents feedback loop)
        self.left_panel.pack_propagate(False)
        self.right_panel.pack_propagate(False)
        self._resize_job = None
        self.bind("<Configure>", self._on_resize)

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill=tk.X)

        self.btn_open = ttk.Button(top, text="Open Image", command=self.on_open)
        self.btn_open.pack(side=tk.LEFT)

        self.status = ttk.Label(top, text="Choose an image (.jpg/.png)")
        self.status.pack(side=tk.LEFT, padx=12)

        # image panes
        mid = ttk.Frame(self, padding=10)
        mid.pack(fill=tk.BOTH, expand=True)

        self.left_panel = ttk.LabelFrame(mid, text="Original")
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.left_img = ttk.Label(self.left_panel, anchor="center")
        self.left_img.pack(fill=tk.BOTH, expand=True)

        self.right_panel = ttk.LabelFrame(mid, text="Grad-CAM")
        self.right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.right_img = ttk.Label(self.right_panel, anchor="center")
        self.right_img.pack(fill=tk.BOTH, expand=True)

    def on_open(self):
        path = filedialog.askopenfilename(
            title="Select MRI Image",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png"),
                       ("All files", "*.*")]
        )
        if not path:
            return
        try:
            pil_img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Open Error", f"Could not open image: {e}")
            return
        
        # run prediction
        try:
            pred_idx, pred_label = predict(self.model, pil_img)
        except Exception as e:
            messagebox.showerror("Inference Error", str(e))
            return

        if pred_label == "notumor":
            self.status.configure(text="No tumor detected.")
            self.right_panel.configure(text="Grad-CAM (n/a)")
            self.right_pil = None
            self.left_pil = pil_img
        else:
            self.status.configure(text="Tumor detected.")
            self.right_panel.configure(text="Grad-CAM")
            try:
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                buf.seek(0)
                result = generate_gradcam(
                    model=self.model,
                    image_path=buf,
                    class_names=LABEL_NAMES,
                    device=DEVICE,
                    return_images=True
                )
                self.left_pil = result["base_img"]
                self.right_pil = result["overlay"]
            except Exception as e:
                messagebox.showerror("Grad-CAM Error", str(e))
                self.right_pil = None
                self.left_pil = pil_img

        self._render_panels()

    def _on_resize(self, _event=None):
        if self._resize_job is not None:
            try:
                self.after_cancel(self._resize_job)
            except Exception:
                pass
        self._resize_job = self.after(60, self._render_panels)


    def _resize_to_fit(self, pil_img: Image.Image, target_w: int, target_h: int) -> Image.Image:
        """
        Resize proportionally to fit inside target box
        """
        if pil_img is None or target_w < 2 or target_h < 2:
            return pil_img
        w0, h0 = pil_img.width, pil_img.height
        scale = min(target_w / max(1, w0), target_h / max(1, h0))
        new_w = max(1, int(round(w0 * scale)))
        new_h = max(1, int(round(h0 * scale)))
        img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        bg = Image.new("RGB", (target_w, target_h), "black")
        off = ((target_w - new_w) // 2, (target_h - new_h) // 2)
        bg.paste(img, off)
        return bg

    def _render_panels(self):
        if self.left_pil is not None:
            lw = max(self.left_panel.winfo_width(), 200)
            lh = max(self.left_panel.winfo_height(), 200)
            left_img = self._resize_to_fit(self.left_pil, lw, lh)
            if left_img is not None:
                ltk = ImageTk.PhotoImage(left_img)
                self.left_img.configure(image=ltk)
                self.left_img.image = ltk

        if self.right_pil is not None:
            rw = max(self.right_panel.winfo_width(), 200)
            rh = max(self.right_panel.winfo_height(), 200)
            right_img = self._resize_to_fit(self.right_pil, rw, rh)
            if right_img is not None:
                rtk = ImageTk.PhotoImage(right_img)
                self.right_img.configure(image=rtk)
                self.right_img.image = rtk
        else:
            self.right_img.configure(image='')
            self.right_img.image = None


if __name__ == "__main__":
    app = TumorApp()
    app.mainloop()