"""Pretrained image encoders that turn area crops into one embedding vector each.

Each backbone is used frozen (no training) and returns float32 vectors. Add a model by adding an
entry to BACKBONES.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class Backbone:
    name: str
    hf_id: str
    kind: str  # "siglip", "hf_pooler" (DINO-style CLS/pooler output) or "radio"
    batch_size: int


BACKBONES = {
    "siglip2": Backbone("siglip2", "google/siglip2-base-patch16-256", "siglip", 64),
    # Open fallback for users without access to DINOv3 (models/dinov2/ is built with it).
    "dinov2": Backbone("dinov2", "facebook/dinov2-base", "hf_pooler", 64),
    # Gated on Hugging Face (Meta approves access manually); needs `hf auth login` once granted.
    "dinov3": Backbone("dinov3", "facebook/dinov3-vitb16-pretrain-lvd1689m", "hf_pooler", 64),
    "radio": Backbone("radio", "nvidia/C-RADIOv4-SO400M", "radio", 16),
}


class Encoder:
    """Loads a backbone onto the GPU (if available) and embeds lists of PIL images."""

    def __init__(self, name: str, device: str | None = None):
        from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPImageProcessor

        self.spec = BACKBONES[name]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        if self.spec.kind == "siglip":
            self.processor = AutoProcessor.from_pretrained(self.spec.hf_id)
            self.model = AutoModel.from_pretrained(self.spec.hf_id, dtype=dtype)
        elif self.spec.kind == "hf_pooler":
            self.processor = AutoImageProcessor.from_pretrained(self.spec.hf_id)
            self.model = AutoModel.from_pretrained(self.spec.hf_id, dtype=dtype)
        else:  # radio: NVIDIA's own model code from the Hugging Face repo
            self.processor = CLIPImageProcessor.from_pretrained(self.spec.hf_id)
            self.model = AutoModel.from_pretrained(self.spec.hf_id, trust_remote_code=True)
        self.model = self.model.to(self.device).eval()
        self.dtype = dtype

    @torch.inference_mode()
    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        kind = self.spec.kind
        if kind == "siglip":
            px = self.processor(images=images, return_tensors="pt")["pixel_values"]
            out = self.model.get_image_features(pixel_values=px.to(self.device, self.dtype)).pooler_output
        elif kind == "hf_pooler":
            # No centre crop: the whole letterboxed crop is the area, so keep all of it.
            px = self.processor(images=images, return_tensors="pt", do_center_crop=False)["pixel_values"]
            out = self.model(pixel_values=px.to(self.device, self.dtype)).pooler_output
        else:
            px = self.processor(images=images, return_tensors="pt", do_resize=True).pixel_values
            with torch.autocast(self.device, dtype=self.dtype, enabled=self.device == "cuda"):
                out, _spatial = self.model(px.to(self.device))
        return out.float().cpu().numpy()

    @torch.inference_mode()
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """SigLIP only: text vectors in the same space as the image vectors."""
        if self.spec.kind != "siglip":
            raise ValueError(f"{self.spec.name} has no text encoder")
        tokens = self.processor(text=texts, return_tensors="pt", padding="max_length", max_length=64)
        out = self.model.get_text_features(**{k: v.to(self.device) for k, v in tokens.items()}).pooler_output
        return out.float().cpu().numpy()


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between every row of a and every row of b."""
    a = F.normalize(torch.from_numpy(a), dim=-1)
    b = F.normalize(torch.from_numpy(b), dim=-1)
    return (a @ b.T).numpy()
