#!/usr/bin/env python
"""Exercise every Olympus routing token and report which produce artifacts.

    python scripts/smoke_test_tokens.py --output-dir outputs/smoke [--fast]
                                        [--only image_gen,video_gen]

Writes ``smoke_results.json`` plus every generated artifact so the results can be
inspected by hand.
"""

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw  # noqa: E402

from olympus_tools.parser import parse  # noqa: E402
from olympus_tools.runner import Runner  # noqa: E402
import olympus_tools.backends as _backends  # noqa: E402

# Must run BEFORE ALL_TASKS is read below: it may repoint 3D_gen_image /
# 3D_gen_text at a fallback backend, and TaskSpec.backend is read from this
# list, not re-resolved per step.
_backends.resolve_3d_backends()

from olympus_tools.tokens import ALL_TASKS, IMAGE, VIDEO, NONE  # noqa: E402

# A short, representative payload per token (what the router would emit).
PROMPTS = {
    "image_gen": "a fluffy orange cat lounging on a windowsill, warm sunlight",
    "image_edit": "make the cat white",
    "image_deblur": "remove the blur and make the image sharp",
    "image_denoise": "clean up the noise in this photo",
    "image_derain": "remove the rain streaks",
    "image_sr": "increase the resolution of this picture",
    "image_det": "detect all objects in the image",
    "image_seg": "segment every object in the picture",
    "image_ground": "the cat",
    "image_depth": "estimate the depth map",
    "image_normal": "estimate the surface normals",
    "image_canny": "extract the canny edges",
    "image_pose": "estimate the human pose",
    "video_gen": "a cat running through a sunlit forest",
    "video_edit": "make it look like a watercolour painting",
    "video_ref_seg": "the cat",
    "3D_gen_text": "a wooden rocking chair",
    "3D_gen_image": "a high-resolution 3D model of this object",
}
for _c in ("pose", "canny", "depth", "normal", "seg", "scrib"):
    PROMPTS[f"{_c}_to_image"] = "a majestic castle at golden hour"
    PROMPTS[f"{_c}_to_video"] = "a majestic castle at golden hour"

# Cheap settings so the sweep finishes in minutes rather than hours.
FAST = {
    "image_gen": {"steps": 12},
    "image_edit": {"steps": 12},
    "video_gen": {"steps": 12, "num_frames": 9},
    "video_edit": {"steps": 8, "max_frames": 6},
    "video_ref_seg": {"max_frames": 4},
    "3D_gen_text": {"steps": 24},
    "3D_gen_image": {"steps": 24},
}
for _c in ("pose", "canny", "depth", "normal", "seg", "scrib"):
    FAST[f"{_c}_to_image"] = {"steps": 12}
    FAST[f"{_c}_to_video"] = {"steps": 8, "num_frames": 4}


def make_seed_image(path: str) -> str:
    """A synthetic photo-ish scene, so the sweep needs no external assets."""
    img = Image.new("RGB", (512, 512), (135, 180, 225))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 340, 512, 512], fill=(110, 145, 90))       # ground
    d.rectangle([150, 170, 360, 350], fill=(190, 175, 150))    # building
    d.polygon([(140, 175), (255, 90), (370, 175)], fill=(140, 70, 60))  # roof
    d.rectangle([215, 250, 285, 350], fill=(90, 65, 50))       # door
    d.ellipse([60, 55, 130, 125], fill=(250, 240, 160))        # sun
    for x in (60, 430):
        d.rectangle([x - 8, 250, x + 8, 350], fill=(95, 70, 45))
        d.ellipse([x - 45, 185, x + 45, 275], fill=(70, 120, 65))
    img.save(path)
    return path


def make_seed_video(path: str, frames: int = 8) -> str:
    from olympus_tools.media import write_video

    imgs = []
    for i in range(frames):
        img = Image.new("RGB", (384, 384), (140, 185, 230))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 260, 384, 384], fill=(110, 145, 90))
        x = 40 + i * 30
        d.ellipse([x, 190, x + 70, 260], fill=(235, 160, 70))  # moving blob
        imgs.append(img)
    write_video(imgs, path, fps=6)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="outputs/smoke")
    ap.add_argument("--fast", action="store_true", help="use reduced step counts")
    ap.add_argument("--only", default=None, help="comma-separated token subset")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seed_img = make_seed_image(os.path.join(args.output_dir, "_seed.png"))
    seed_vid = make_seed_video(os.path.join(args.output_dir, "_seed.mp4"))

    wanted = set(args.only.split(",")) if args.only else None
    tasks = [t for t in ALL_TASKS if wanted is None or t.token in wanted]

    results, t_start = [], time.time()
    for spec in tasks:
        token = spec.token
        out_dir = os.path.join(args.output_dir, token)
        os.makedirs(out_dir, exist_ok=True)
        payload = PROMPTS.get(token, "test")
        plan = parse(f"<{token}>{payload}</{token}>", instruction=f"smoke:{token}")

        opts = {token: FAST.get(token, {})} if args.fast else {}
        runner = Runner(out_dir, device=args.device, step_options=opts)
        t0 = time.time()
        record = {"token": token, "backend": spec.backend}
        try:
            manifest = runner.run(
                plan,
                user_image=seed_img if spec.consumes in (IMAGE, VIDEO) else None,
                user_video=seed_vid if spec.consumes == VIDEO else None,
            )
            step = manifest["steps"][0]
            record["status"] = step["status"]
            record["outputs"] = step.get("outputs", {})
            record["error"] = step.get("error")
            record["model"] = step.get("model")
        except Exception as exc:
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc(limit=3)
        finally:
            runner.close()
        record["seconds"] = round(time.time() - t0, 1)
        results.append(record)
        flag = "ok " if record["status"] == "ok" else "FAIL"
        print(f"[{flag}] <{token}>  {record['seconds']}s  {record.get('error') or ''}",
              flush=True)

    path = os.path.join(args.output_dir, "smoke_results.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)

    ok = [r for r in results if r["status"] == "ok"]
    print(f"\n=== {len(ok)}/{len(results)} tokens produced artifacts "
          f"in {round(time.time() - t_start)}s ===")
    for r in results:
        if r["status"] != "ok":
            print(f"  FAILED <{r['token']}>: {r['error']}")
    print(f"results: {path}")


if __name__ == "__main__":
    main()
