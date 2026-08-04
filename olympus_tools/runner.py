"""Execute a parsed :class:`~olympus_tools.parser.Plan` against the specialists."""

import json
import os
import time
import traceback
from typing import Dict, Optional

from .backends.base import ModelHost
from .parser import Plan, Step, USER_INPUT
from .tokens import IMAGE, VIDEO, MESH, ANNOTATION, NONE

# Estimation tokens whose output already *is* a ControlNet condition map, so a
# chained <cond>_to_image / <cond>_to_video step must not re-derive it.
_MAP_PRODUCERS = {
    "image_canny": "canny",
    "image_depth": "depth",
    "image_normal": "normal",
    "image_pose": "pose",
    "image_seg": "seg",
}


class Runner:
    def __init__(self, output_dir: str, device: str = "cuda", dtype: str = "fp16",
                 overrides: Optional[Dict[str, dict]] = None,
                 max_resident: int = 1, step_options: Optional[Dict[str, dict]] = None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.host = ModelHost(device=device, dtype=dtype,
                              max_resident=max_resident, overrides=overrides)
        self.step_options = step_options or {}

    # -- artifact plumbing -------------------------------------------------
    @staticmethod
    def _pick(result: dict, kind: str) -> Optional[str]:
        if kind == IMAGE:
            return result.get("image")
        if kind == VIDEO:
            return result.get("video") or result.get("image")
        if kind == MESH:
            return result.get("mesh")
        if kind == ANNOTATION:
            return result.get("annotation") or result.get("image")
        return None

    def _resolve_input(self, step: Step, results, user_image, user_video):
        if step.consumes == NONE:
            return None, False
        if step.input_from == USER_INPUT:
            src = user_video if step.consumes == VIDEO else user_image
            if src is None and step.consumes == VIDEO:
                src = user_image  # a still can seed Text2Video-Zero
            if src is None:
                raise ValueError(
                    f"<{step.token}> consumes a{'n' if step.consumes[0] in 'aeiou' else ''} "
                    f"{step.consumes} but none was produced upstream; "
                    f"pass --input-{'video' if step.consumes == VIDEO else 'image'}"
                )
            return src, False

        prev = results.get(step.input_from)
        if prev is None or "error" in prev:
            raise ValueError(f"<{step.token}> depends on step {step.input_from}, "
                             f"which did not produce an artifact")
        path = self._pick(prev, step.consumes) or self._pick(prev, IMAGE)
        if path is None:
            raise ValueError(f"step {step.input_from} produced no reusable "
                             f"{step.consumes} artifact")

        # Is the upstream artifact already the condition map this step needs?
        already_map = False
        if step.condition:
            producer_token = prev.get("_token")
            already_map = _MAP_PRODUCERS.get(producer_token) == step.condition
        return path, already_map

    # -- main loop ---------------------------------------------------------
    def run(self, plan: Plan, user_image: Optional[str] = None,
            user_video: Optional[str] = None, dry_run: bool = False) -> dict:
        results: Dict[int, dict] = {}
        manifest = {
            "instruction": plan.instruction,
            "router_output": plan.raw_output,
            "direct_answer": plan.direct_answer,
            "steps": [],
        }

        for step in plan.steps:
            stem = os.path.join(self.output_dir, f"step{step.index}_{step.token}")
            entry = {
                "index": step.index,
                "token": step.token,
                "task": step.task,
                "prompt": step.prompt,
                "backend": step.backend,
                "paper_model": step.paper_model,
                "input_from": step.input_from,
            }
            try:
                src, already_map = self._resolve_input(step, results,
                                                       user_image, user_video)
                entry["input"] = src
                if dry_run:
                    entry["status"] = "planned"
                    manifest["steps"].append(entry)
                    results[step.index] = {"_token": step.token, "image": f"{stem}.png"}
                    continue

                backend = self.host.get(step.backend)
                opts = dict(self.step_options.get(step.token, {}))
                opts["already_map"] = already_map
                t0 = time.time()
                out = backend.run(step.prompt, src, stem, step=step, **opts)
                entry["seconds"] = round(time.time() - t0, 1)
                entry["outputs"] = {k: v for k, v in out.items()
                                    if isinstance(v, (str, int, float, list))}
                entry["status"] = "ok"
                if backend.substitution:
                    entry["substitution"] = backend.substitution
                out["_token"] = step.token
                results[step.index] = out
                print(f"  [{step.index}] <{step.token}> -> "
                      f"{', '.join(str(v) for k, v in out.items() if k != '_token')}")
            except Exception as exc:
                entry["status"] = "error"
                entry["error"] = f"{type(exc).__name__}: {exc}"
                entry["traceback"] = traceback.format_exc(limit=3)
                results[step.index] = {"error": str(exc), "_token": step.token}
                print(f"  [{step.index}] <{step.token}> FAILED: {exc}")
            manifest["steps"].append(entry)

        path = os.path.join(self.output_dir, "manifest.json")
        with open(path, "w") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)
        manifest["manifest_path"] = path
        return manifest

    def close(self):
        self.host.shutdown()
