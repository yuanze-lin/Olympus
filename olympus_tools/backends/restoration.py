"""Restoration specialists: deblurring, denoising, deraining, super-resolution.

Table 9 assigns InstructIR to ``<image_deblur>``, ``<image_denoise>`` and
``<image_derain>``, and Swin2SR to ``<image_sr>``. Both are used as specified.
InstructIR ships as a GitHub repo rather than a pip package, so it is cloned into
``third_party/InstructIR`` by ``scripts/install_specialists.sh``.
"""

import os
import sys
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .base import Backend, register, hf_kwargs

# Prompts steering InstructIR's language head per restoration token.
INSTRUCTIR_PROMPTS = {
    "image_deblur": "Remove the blur and make the image sharp and clear",
    "image_denoise": "Clean up the noise and grain in this photo",
    "image_derain": "Remove the rain streaks so the picture looks clear",
}


def _instructir_dir() -> Optional[str]:
    cand = os.environ.get("OLYMPUS_INSTRUCTIR_DIR")
    if cand and os.path.isdir(cand):
        return cand
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cand = os.path.join(here, "third_party", "InstructIR")
    return cand if os.path.isdir(cand) else None


@register("instructir")
class InstructIRBackend(Backend):
    """<image_deblur> / <image_denoise> / <image_derain> -- InstructIR (Table 9).

    One all-in-one restoration model steered by a natural-language instruction,
    so the same weights serve all three routing tokens; the instruction is taken
    from the routing-token payload the router produced.
    """

    default_model_id = "marcosv/InstructIR"

    def load(self):
        repo = _instructir_dir()
        if repo is None:
            raise RuntimeError(
                "InstructIR is not installed. Run scripts/install_specialists.sh "
                "or set OLYMPUS_INSTRUCTIR_DIR to a clone of "
                "https://github.com/mv-lab/InstructIR"
            )
        if repo not in sys.path:
            sys.path.insert(0, repo)

        import yaml
        from huggingface_hub import hf_hub_download
        from models import instructir  # noqa: E402  (from the cloned repo)
        from text.models import LanguageModel, LMHead  # noqa: E402

        with open(os.path.join(repo, "configs", "eval5d.yml")) as fh:
            cfg = yaml.safe_load(fh)
        m, llm = cfg["model"], cfg["llm"]

        # Prefer the checkpoints shipped in the repo; fall back to the Hub.
        def _weights(name):
            local = os.path.join(repo, "models", name)
            return local if os.path.exists(local) else hf_hub_download(self.model_id, name)

        self.model = instructir.create_model(
            input_channels=m["in_ch"], width=m["width"], enc_blks=m["enc_blks"],
            middle_blk_num=m["middle_blk_num"], dec_blks=m["dec_blks"],
            txtdim=m["textdim"],
        )
        self.model.load_state_dict(torch.load(_weights("im_instructir-7d.pt"),
                                              map_location="cpu"), strict=True)
        self.model = self.model.to(self.device).eval()

        self.lm = LanguageModel(model=llm["model"])
        self.lm_head = LMHead(embedding_dim=llm["model_dim"],
                              hidden_dim=llm["embd_dim"], num_classes=llm["nclasses"])
        self.lm_head.load_state_dict(torch.load(_weights("lm_instructir-7d.pt"),
                                                map_location="cpu"), strict=True)
        self.lm_head = self.lm_head.to(self.device).eval()

    def run(self, prompt, input_path, out_stem, step=None, **kw):
        if not input_path:
            raise ValueError("InstructIR tasks need an input image")
        token = step.token if step else "image_denoise"
        instruction = prompt or INSTRUCTIR_PROMPTS.get(token, "Restore this image")

        img = Image.open(input_path).convert("RGB")
        x = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            lm_embd = self.lm([instruction]).to(self.device)
            text_embd, _ = self.lm_head(lm_embd)
            y = self.model(x, text_embd).clamp(0, 1)

        arr = (y[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        out = f"{out_stem}.png"
        Image.fromarray(arr).save(out)
        return {"image": out, "instruction": instruction}


@register("swin2sr")
class Swin2SRBackend(Backend):
    """<image_sr> -- Swin2SR (Table 9, exact)."""

    default_model_id = "caidas/swin2SR-classical-sr-x2-64"

    def load(self):
        from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = Swin2SRForImageSuperResolution.from_pretrained(
            self.model_id).to(self.device).eval()

    def run(self, prompt, input_path, out_stem, step=None, **kw):
        if not input_path:
            raise ValueError("<image_sr> needs an input image")
        img = Image.open(input_path).convert("RGB")
        # Swin2SR is memory-hungry; cap the input so 2x stays on one GPU.
        cap = kw.get("max_side", 512)
        if max(img.size) > cap:
            img.thumbnail((cap, cap))
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_t = self.model(**inputs).reconstruction
        arr = out_t.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        arr = (arr * 255).round().astype(np.uint8)
        out = f"{out_stem}.png"
        Image.fromarray(arr).save(out)
        return {"image": out, "size": Image.fromarray(arr).size}
