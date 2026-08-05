#!/usr/bin/env python
"""Run Olympus end to end: instruction -> routing tokens -> specialist models.

This is the piece that turns Olympus's routing tokens into actual images, videos
and 3D meshes (GitHub issue #1).

Examples
--------
Route an instruction and execute every task it dispatches::

    python run_tools.py --prompt "Generate an image of a fluffy orange cat ..." \\
        --model-path ckpts/Olympus --output-dir outputs/cat

Only show the plan (no GPU, no weights downloaded)::

    python run_tools.py --prompt "..." --dry-run

Execute a plan produced earlier (or hand-written), skipping the router::

    python run_tools.py --plan outputs/cat/plan.json --output-dir outputs/cat
"""

import argparse
import json
import os
import sys

from olympus_tools.parser import parse, Plan


def route(args) -> Plan:
    """Run the Olympus router to turn an instruction into routing tokens."""
    import torch

    from mipha.constants import IMAGE_TOKEN_INDEX
    from mipha.conversation import conv_templates
    from mipha.model.builder import load_pretrained_model
    from mipha.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                                tokenizer_image_token)
    from mipha.utils import disable_torch_init

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    tokenizer, model, _, _ = load_pretrained_model(
        model_path, args.model_base, get_model_name_from_path(model_path)
    )

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], args.prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()
    stop_token = conv.sep2
    criteria = [KeywordsStoppingCriteria([stop_token], tokenizer, input_ids)]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            images=None,
            do_sample=(args.temperature > 0),
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            stopping_criteria=criteria,
        )
    decoded = tokenizer.batch_decode(
        output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
    )[0].strip()
    if decoded.endswith(stop_token):
        decoded = decoded[: -len(stop_token)].strip()

    # Free the router before any specialist is built -- they share one GPU.
    del model
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return parse(decoded, instruction=args.prompt)


def _parse_kv(pairs, cast_json=False):
    """Parse ``key=value`` CLI pairs into a dict."""
    out = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"expected key=value, got '{item}'")
        k, v = item.split("=", 1)
        if cast_json:
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                pass
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Execute Olympus routing tokens with specialist models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Not required at parse time so that --list-tokens works on its own.
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--prompt", type=str, help="user instruction to route")
    src.add_argument("--plan", type=str, help="path to a saved plan.json")
    src.add_argument("--router-output", type=str,
                     help="raw routing-token string, bypassing the router")

    ap.add_argument("--model-path", type=str, default="ckpts/Olympus")
    ap.add_argument("--model-base", type=str, default=None)
    ap.add_argument("--conv-mode", type=str, default="v0")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=1024)

    ap.add_argument("--output-dir", type=str, default="outputs/olympus_run")
    ap.add_argument("--input-image", type=str, default=None,
                    help="image consumed by the first image-consuming task")
    ap.add_argument("--input-video", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--max-resident", type=int, default=1,
                    help="how many specialists may stay in VRAM at once")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan without loading any specialist")
    ap.add_argument("--backend-model", nargs="*", default=None, metavar="BACKEND=HF_ID",
                    help="override a backend checkpoint, e.g. detection=<hf-id>")
    ap.add_argument("--step-option", nargs="*", default=None, metavar="TOKEN.KEY=VALUE",
                    help="per-token knob, e.g. video_gen.num_frames=25")
    ap.add_argument("--list-tokens", action="store_true",
                    help="print every supported routing token and exit")
    ap.add_argument("--legacy-backends", action="store_true",
                    help="use the paper's Table 9 specialists (SDXL, InstructPix2Pix, "
                         "CogVideoX, Text2Video-Zero, Depth Anything V2) instead of "
                         "the current SOTA defaults")
    args = ap.parse_args()

    # Backend selection must settle BEFORE the plan is parsed, because each step
    # records the backend it will run on.
    import olympus_tools.backends as _backends  # registers every backend

    if args.legacy_backends:
        from olympus_tools.tokens import apply_legacy_backends

        apply_legacy_backends()
        print("Using the paper's Table 9 specialists (--legacy-backends).")
    else:
        _backends.resolve_3d_backends()  # downgrade 3D tokens if TRELLIS.2 is absent

    if args.list_tokens:
        from olympus_tools.tokens import ALL_TASKS

        print(f"{'ROUTING TOKEN':<22} {'TASK':<42} {'BACKEND':<20} PAPER (Table 9)")
        for spec in ALL_TASKS:
            print(f"{spec.open_tag:<22} {spec.task:<42} {spec.backend}")
        return

    if not (args.prompt or args.plan or args.router_output):
        ap.error("one of --prompt, --plan or --router-output is required")

    # ---- 1. obtain a plan ------------------------------------------------
    if args.plan:
        plan = Plan.from_json(open(args.plan).read())
    elif args.router_output:
        plan = parse(args.router_output, instruction="(router output supplied)")
    else:
        print(f"Routing instruction through Olympus ({args.model_path}) ...")
        plan = route(args)

    print("\nRouter output:\n ", plan.routed_output or "(none)")
    if plan.direct_answer:
        print("\nDirect answer:\n ", plan.direct_answer)
    if plan.unknown_tokens:
        print("\n[warn] unrecognised tokens:", ", ".join(plan.unknown_tokens))
    print("\nExecution plan:")
    print(plan.describe())

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "plan.json"), "w") as fh:
        fh.write(plan.to_json())

    if not plan.steps:
        print("\nNo routing tokens -- nothing to dispatch.")
        return

    # ---- 2. execute ------------------------------------------------------
    overrides = {k: {"model_id": v} for k, v in _parse_kv(args.backend_model).items()}
    step_options = {}
    for key, value in _parse_kv(args.step_option, cast_json=True).items():
        token, _, opt = key.partition(".")
        if not opt:
            raise SystemExit(f"--step-option expects TOKEN.KEY=VALUE, got '{key}'")
        step_options.setdefault(token, {})[opt] = value

    from olympus_tools.runner import Runner
    import olympus_tools.backends as _backends  # registers backends


    print(f"\nExecuting {len(plan.steps)} step(s) -> {args.output_dir}")
    runner = Runner(args.output_dir, device=args.device, dtype=args.dtype,
                    overrides=overrides, max_resident=args.max_resident,
                    step_options=step_options)
    try:
        manifest = runner.run(plan, user_image=args.input_image,
                              user_video=args.input_video, dry_run=args.dry_run)
    finally:
        runner.close()

    if args.dry_run:
        print(f"\nDry run: {len(manifest['steps'])} step(s) planned, "
              f"no specialist was loaded.")
    else:
        ok = sum(1 for s in manifest["steps"] if s["status"] == "ok")
        print(f"\nDone: {ok}/{len(manifest['steps'])} step(s) succeeded.")
    print(f"Manifest: {manifest['manifest_path']}")


if __name__ == "__main__":
    sys.exit(main())
