#!/usr/bin/env python
"""Pre-download every specialist checkpoint used by run_tools.py.

Diffusers pipelines are fetched through ``DiffusionPipeline.download`` so only
the files a pipeline actually loads come down (skipping fp32 duplicates and the
monolithic single-file checkpoints some repos also publish). Plain transformers
models use ``snapshot_download``.
"""
import sys
import traceback

from huggingface_hub import snapshot_download

# Diffusers pipeline repos: (repo_id, preferred variant)
PIPELINES = [
    ("stabilityai/stable-diffusion-xl-base-1.0", "fp16"),
    ("timbrooks/instruct-pix2pix", "fp16"),
    ("stable-diffusion-v1-5/stable-diffusion-v1-5", "fp16"),
    ("THUDM/CogVideoX-2b", None),
    ("openai/shap-e", None),
    ("openai/shap-e-img2img", None),
]

# ControlNet weights load as a bare model, not a pipeline.
CONTROLNETS = [
    "lllyasviel/control_v11p_sd15_canny",
    "lllyasviel/control_v11f1p_sd15_depth",
    "lllyasviel/control_v11p_sd15_normalbae",
    "lllyasviel/control_v11p_sd15_openpose",
    "lllyasviel/control_v11p_sd15_seg",
    "lllyasviel/control_v11p_sd15_scribble",
]

# transformers / misc repos: (repo_id, allow_patterns, ignore_patterns)
PLAIN = [
    ("nvidia/segformer-b5-finetuned-ade-640-640", None, None),
    ("IDEA-Research/grounding-dino-base", None, ["*.bin"]),
    ("facebook/detr-resnet-101", None, ["*.bin"]),
    ("depth-anything/Depth-Anything-V2-Small-hf", None, ["*.bin"]),
    ("caidas/swin2SR-classical-sr-x2-64", None, ["*.bin"]),
    ("facebook/sam-vit-base", None, ["*.bin"]),
    ("lllyasviel/Annotators", ["body_pose_model.pth", "hand_pose_model.pth",
                               "facenet.pth", "scannet.pt", "table5_pidinet.pth"], None),
    ("marcosv/InstructIR", ["*.pt"], None),
]

failed = []


def attempt(label, fn):
    try:
        print(f"--- {label}", flush=True)
        fn()
    except Exception as exc:
        print(f"!!! FAILED {label}: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        failed.append(label)


def main():
    from diffusers import DiffusionPipeline

    for repo, variant in PIPELINES:
        def _dl(repo=repo, variant=variant):
            # Shap-E only publishes .bin weights, so safetensors must not be forced.
            safet = not repo.startswith("openai/shap-e")
            try:
                DiffusionPipeline.download(repo, variant=variant,
                                           use_safetensors=safet, max_workers=8)
            except Exception:
                # not every repo publishes an fp16 variant
                DiffusionPipeline.download(repo, use_safetensors=safet, max_workers=8)
        attempt(repo, _dl)

    for repo in CONTROLNETS:
        attempt(repo, lambda repo=repo: snapshot_download(
            repo_id=repo, allow_patterns=["*.json", "*.safetensors"], max_workers=8))

    for repo, allow, ignore in PLAIN:
        attempt(repo, lambda repo=repo, allow=allow, ignore=ignore: snapshot_download(
            repo_id=repo, allow_patterns=allow, ignore_patterns=ignore, max_workers=8))

    print("\n=== PREFETCH COMPLETE ===")
    print("failed:", failed if failed else "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
