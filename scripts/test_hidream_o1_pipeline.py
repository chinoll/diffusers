# Copyright 2026 chinoll and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys

import torch
from transformers import AutoProcessor


SOURCE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE_SRC = os.path.join(SOURCE_ROOT, "src")
if os.path.exists(os.path.join(SOURCE_SRC, "diffusers", "__init__.py")):
    sys.path.insert(0, SOURCE_SRC)

from diffusers import FlowMatchEulerDiscreteScheduler, HiDreamO1ImagePipeline, UniPCMultistepScheduler


DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test HiDream-O1-Image with Diffusers.")
    parser.add_argument("--model-id", default="HiDream-ai/HiDream-O1-Image")
    parser.add_argument(
        "--official-repo-path",
        default=None,
        help=(
            "Path to a cloned HiDream-O1-Image repo. If set, the script imports "
            "`models.qwen3_vl_transformers.Qwen3VLForConditionalGeneration` from it."
        ),
    )
    parser.add_argument("--prompt", default="A cinematic portrait of a woman in candlelight.")
    parser.add_argument("--ref-images", nargs="*", default=None)
    parser.add_argument("--layout-bboxes", default=None)
    parser.add_argument("--output", default="hidream_o1_test.png")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--model-type", choices=["full", "dev"], default="dev")
    parser.add_argument("--scheduler-type", choices=["auto", "default", "flow_match", "flash"], default="auto")
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--shift", type=float, default=None)
    parser.add_argument("--noise-scale-start", type=float, default=7.5)
    parser.add_argument("--noise-scale-end", type=float, default=7.5)
    parser.add_argument("--noise-clip-std", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--device-map", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--generator-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--keep-original-aspect", action="store_true")
    parser.add_argument(
        "--use-resolution-binning",
        action="store_true",
        help="Snap width and height to the official HiDream-O1 resolution bins.",
    )
    parser.add_argument("--disable-flash-attn", action="store_true")
    return parser.parse_args()


def compact_kwargs(**kwargs):
    return {key: value for key, value in kwargs.items() if value is not None}


def load_transformer(args, dtype):
    pretrained_kwargs = compact_kwargs(
        torch_dtype=dtype,
        device_map=args.device_map,
        cache_dir=args.cache_dir,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )

    if args.official_repo_path is not None:
        sys.path.insert(0, args.official_repo_path)
        from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration.from_pretrained(args.model_id, **pretrained_kwargs).eval()

    import transformers

    errors = []
    for class_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        model_cls = getattr(transformers, class_name, None)
        if model_cls is None:
            continue
        try:
            return model_cls.from_pretrained(args.model_id, **pretrained_kwargs).eval()
        except Exception as error:
            errors.append(f"{class_name}: {error}")

    joined_errors = "\n".join(errors) if errors else "No compatible AutoModel class was found."
    raise RuntimeError(
        "Could not load the HiDream-O1 transformer from transformers.\n"
        "Pass --official-repo-path /path/to/HiDream-O1-Image, or install a transformers version that supports this "
        f"architecture.\n{joined_errors}"
    )


def resolve_scheduler_type(args):
    if args.scheduler_type != "auto":
        return args.scheduler_type
    if args.model_type == "full":
        return "default"
    if args.ref_images is not None and len(args.ref_images) == 1:
        return "flow_match"
    return "flash"


def build_scheduler(scheduler_type, shift):
    if scheduler_type == "default":
        return UniPCMultistepScheduler(
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            flow_shift=shift,
        )
    return FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=shift,
    )


def main():
    args = parse_args()
    dtype = DTYPES[args.dtype]
    scheduler_type = resolve_scheduler_type(args)
    shift = args.shift if args.shift is not None else 3.0 if args.model_type == "full" else 1.0

    processor_kwargs = compact_kwargs(
        cache_dir=args.cache_dir,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )

    processor = AutoProcessor.from_pretrained(args.model_id, **processor_kwargs)
    transformer = load_transformer(args, dtype=dtype)
    scheduler = build_scheduler(scheduler_type, shift=shift)

    pipe = HiDreamO1ImagePipeline(
        transformer=transformer,
        processor=processor,
        scheduler=scheduler,
    )
    pipe.set_progress_bar_config(disable=False)

    generator = torch.Generator(device=args.generator_device).manual_seed(args.seed)
    output = pipe(
        prompt=args.prompt,
        ref_images=args.ref_images,
        height=args.height,
        width=args.width,
        model_type=args.model_type,
        scheduler_type=scheduler_type,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        shift=shift,
        noise_scale_start=args.noise_scale_start,
        noise_scale_end=args.noise_scale_end,
        noise_clip_std=args.noise_clip_std,
        keep_original_aspect=args.keep_original_aspect,
        layout_bboxes=args.layout_bboxes,
        use_resolution_binning=args.use_resolution_binning,
        use_flash_attn=not args.disable_flash_attn,
        generator=generator,
        output_type="pil",
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    output.images[0].save(args.output)
    print(f"Saved image to {args.output}")


if __name__ == "__main__":
    main()
