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

import inspect
import json
import math
import os
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import PIL.Image
import torch
from PIL import Image, ImageDraw
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin

from ...image_processor import PipelineImageInput
from ...schedulers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from ...utils import is_torch_xla_available, logging, numpy_to_pil, replace_example_docstring
from ...utils.loading_utils import load_image
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import HiDreamO1ImagePipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from transformers import AutoModelForImageTextToText, AutoProcessor
        >>> from diffusers import HiDreamO1ImagePipeline, UniPCMultistepScheduler

        >>> model_id = "HiDream-ai/HiDream-O1-Image"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> transformer = AutoModelForImageTextToText.from_pretrained(
        ...     model_id, torch_dtype=torch.bfloat16, device_map="cuda"
        ... )
        >>> scheduler = UniPCMultistepScheduler(
        ...     prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0
        ... )
        >>> pipe = HiDreamO1ImagePipeline(transformer=transformer, processor=processor, scheduler=scheduler)
        >>> image = pipe(
        ...     "A cinematic portrait of a woman in candlelight.",
        ...     height=2048,
        ...     width=2048,
        ...     guidance_scale=5.0,
        ...     num_inference_steps=50,
        ... ).images[0]
        >>> image.save("hidream_o1.png")
        ```
"""

TIMESTEP_TOKEN_NUM = 1
DEFAULT_PATCH_SIZE = 32
DEFAULT_CONDITION_IMAGE_SIZE = 384
DEFAULT_NOISE_SCALE = 8.0
T_EPS = 0.001

DEFAULT_TIMESTEPS = [
    999,
    987,
    974,
    960,
    945,
    929,
    913,
    895,
    877,
    857,
    836,
    814,
    790,
    764,
    737,
    707,
    675,
    640,
    602,
    560,
    515,
    464,
    409,
    347,
    278,
    199,
    110,
    8,
]

PREDEFINED_RESOLUTIONS = [
    (2048, 2048),
    (2304, 1728),
    (1728, 2304),
    (2560, 1440),
    (1440, 2560),
    (2496, 1664),
    (1664, 2496),
    (3104, 1312),
    (1312, 3104),
    (2304, 1792),
    (1792, 2304),
]

DEFAULT_COLORS = [
    (255, 0, 0),
    (0, 180, 0),
    (0, 0, 255),
    (204, 180, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
]


def _get_tokenizer(processor):
    if isinstance(processor, PreTrainedTokenizerBase):
        return processor
    return processor.tokenizer


def _add_special_token_shortcuts(tokenizer):
    tokenizer.boi_token = getattr(tokenizer, "boi_token", "<|boi_token|>")
    tokenizer.bor_token = getattr(tokenizer, "bor_token", "<|bor_token|>")
    tokenizer.eor_token = getattr(tokenizer, "eor_token", "<|eor_token|>")
    tokenizer.bot_token = getattr(tokenizer, "bot_token", "<|bot_token|>")
    tokenizer.tms_token = getattr(tokenizer, "tms_token", "<|tms_token|>")


def _find_closest_resolution(width: int, height: int) -> tuple[int, int]:
    image_ratio = width / height
    best_resolution = PREDEFINED_RESOLUTIONS[0]
    min_diff = float("inf")

    for predefined_width, predefined_height in PREDEFINED_RESOLUTIONS:
        diff = abs(predefined_width / predefined_height - image_ratio)
        if diff < min_diff:
            min_diff = diff
            best_resolution = (predefined_width, predefined_height)

    return best_resolution


def _resize_pil_image(
    image: PIL.Image.Image,
    image_size: int,
    patch_size: int = DEFAULT_PATCH_SIZE,
    resampler=Image.BICUBIC,
) -> PIL.Image.Image:
    while min(*image.size) >= 2 * image_size:
        image = image.resize(tuple(x // 2 for x in image.size), resample=Image.BOX)

    multiple = patch_size
    width, height = image.width, image.height
    max_area = image_size * image_size
    scale = math.sqrt(max_area / (width * height))

    new_sizes = [
        (round(width * scale) // multiple * multiple, round(height * scale) // multiple * multiple),
        (round(width * scale) // multiple * multiple, math.floor(height * scale) // multiple * multiple),
        (math.floor(width * scale) // multiple * multiple, round(height * scale) // multiple * multiple),
        (math.floor(width * scale) // multiple * multiple, math.floor(height * scale) // multiple * multiple),
    ]
    new_sizes = sorted(new_sizes, key=lambda x: x[0] * x[1], reverse=True)

    for new_size in new_sizes:
        if new_size[0] > 0 and new_size[1] > 0 and new_size[0] * new_size[1] <= max_area:
            break
    else:
        raise ValueError("Could not compute a positive patch-aligned size for the reference image.")

    width_scale = width / new_size[0]
    height_scale = height / new_size[1]
    if width_scale < height_scale:
        image = image.resize([new_size[0], round(height / width_scale)], resample=resampler)
        top = (round(height / width_scale) - new_size[1]) // 2
        image = image.crop((0, top, new_size[0], top + new_size[1]))
    else:
        image = image.resize([round(width / height_scale), new_size[1]], resample=resampler)
        left = (round(width / height_scale) - new_size[0]) // 2
        image = image.crop((left, 0, left + new_size[0], new_size[1]))

    return image


def _calculate_dimensions(max_size: int, ratio: float) -> tuple[int, int]:
    width = math.sqrt(max_size * max_size * ratio)
    height = width / ratio
    width = int(width / DEFAULT_PATCH_SIZE) * DEFAULT_PATCH_SIZE
    height = int(height / DEFAULT_PATCH_SIZE) * DEFAULT_PATCH_SIZE
    return width, height


def _load_layout_bboxes(layout_bboxes: str | list | dict) -> Any:
    if isinstance(layout_bboxes, str):
        if os.path.exists(layout_bboxes):
            with open(layout_bboxes, encoding="utf-8") as f:
                return json.load(f)
        return json.loads(layout_bboxes)
    return layout_bboxes


def _unwrap_boxes(data: Any) -> Any:
    if isinstance(data, dict):
        for key in ("layout_bboxes", "bboxes", "boxes", "bbox_list"):
            if key in data:
                return data[key]
    return data


def _as_bbox_and_text(item: Any) -> tuple[Sequence[float], str]:
    if isinstance(item, dict):
        bbox = item.get("bbox") or item.get("box")
        text = str(item.get("text") or item.get("label") or "")
        if bbox is None:
            raise ValueError(f"Missing bbox in layout item: {item!r}")
        return bbox, text
    if isinstance(item, (list, tuple)) and len(item) == 4:
        return item, ""
    raise ValueError(f"Unsupported layout bbox item: {item!r}")


def _xxyy_relative_to_absolute_bbox(bbox: Sequence[float], width: int, height: int) -> list[int]:
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox with 4 values, got: {bbox!r}")

    x1, x2, y1, y2 = [float(v) for v in bbox]
    max_abs = max(abs(x1), abs(y1), abs(x2), abs(y2))
    if max_abs <= 1.0:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    elif max_abs <= 100.0:
        x1, x2 = x1 / 100.0 * width, x2 / 100.0 * width
        y1, y2 = y1 / 100.0 * height, y2 / 100.0 * height

    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(0, min(width - 1, int(round(x2))))
    y2 = max(0, min(height - 1, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox after scaling/clamping: {[x1, y1, x2, y2]!r}")
    return [x1, y1, x2, y2]


def _parse_layout_bboxes(layout_bboxes: Any, width: int, height: int) -> list[dict[str, Any]]:
    raw_boxes = _unwrap_boxes(layout_bboxes)
    if not isinstance(raw_boxes, list):
        raise ValueError("`layout_bboxes` must be a list, or a dict containing one of layout_bboxes/bboxes/boxes.")

    parsed = []
    for idx, item in enumerate(raw_boxes):
        bbox, text = _as_bbox_and_text(item)
        parsed.append(
            {
                "bbox": _xxyy_relative_to_absolute_bbox(bbox, width, height),
                "color": "",
                "text": text,
                "image": None,
                "_orig_idx": idx,
            }
        )
    return parsed


def _bbox_area(item: dict[str, Any]) -> int:
    x1, y1, x2, y2 = item["bbox"]
    return max(0, x2 - x1) * max(0, y2 - y1)


def _get_render_params(image_width: int, image_height: int) -> tuple[int, int]:
    edge = math.sqrt(image_width * image_height)
    max_font_size = int(edge * 0.07)
    max_bbox_line_width = int(edge * 0.05)
    return max_font_size, max_bbox_line_width


def _draw_bbox_layout(
    bbox_list: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    max_bbox: int = 5,
    max_bbox_line_width: int | None = None,
    bbox_line_gap: int | None = None,
    return_color: bool = False,
):
    if max_bbox_line_width is None:
        _, max_bbox_line_width = _get_render_params(image_width, image_height)
    if bbox_line_gap is None:
        bbox_line_gap = max(1, max_bbox_line_width // max_bbox)

    image = Image.new("RGB", (image_width, image_height), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    color_list = [None] * len(bbox_list)
    sorted_bboxes = sorted(bbox_list, key=_bbox_area, reverse=True)[:max_bbox]

    for sorted_idx, item in enumerate(sorted_bboxes):
        color = DEFAULT_COLORS[sorted_idx % len(DEFAULT_COLORS)]
        orig_idx = int(item.get("_orig_idx", sorted_idx))
        if 0 <= orig_idx < len(color_list):
            color_list[orig_idx] = color
        line_width = max(max_bbox_line_width - sorted_idx * bbox_line_gap, 5)
        draw.rectangle([int(v) for v in item["bbox"]], outline=color, width=line_width)

    if return_color:
        return image, color_list
    return image


def _add_outer_border_keep_size(image: PIL.Image.Image, color: Iterable[int], width: int) -> PIL.Image.Image:
    image = image.convert("RGB").copy()
    color_tuple = tuple(int(c) for c in color)
    width = max(0, int(width))
    if width == 0:
        return image

    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size
    for offset in range(width):
        draw.rectangle(
            [offset, offset, image_width - 1 - offset, image_height - 1 - offset],
            outline=color_tuple,
        )
    return image


def _create_layout_reference_images(
    ref_images: Sequence[PIL.Image.Image],
    layout_bboxes: Any,
    image_width: int,
    image_height: int,
    ref_max_size: int | None = None,
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> list[PIL.Image.Image]:
    parsed_boxes = _parse_layout_bboxes(layout_bboxes, image_width, image_height)
    layout_image, color_list = _draw_bbox_layout(
        parsed_boxes,
        image_width=image_width,
        image_height=image_height,
        return_color=True,
    )

    output_refs = []
    for idx, ref_image in enumerate(ref_images):
        if ref_max_size is not None:
            ref_image = _resize_pil_image(ref_image, ref_max_size, patch_size)
        color = (
            color_list[idx]
            if idx < len(color_list) and color_list[idx] is not None
            else DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        )
        line_width = int(math.sqrt(ref_image.width * ref_image.height) * 0.04)
        output_refs.append(_add_outer_border_keep_size(ref_image, color, line_width))

    output_refs.append(layout_image)
    return output_refs


def _get_rope_index_fix_point(
    spatial_merge_size,
    image_token_id,
    video_token_id,
    vision_start_token_id,
    input_ids: torch.LongTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    skip_vision_start_token=None,
    fix_point=4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list = []
            start = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    end_image = input_tokens.index(image_token_id, start)
                else:
                    end_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    end_video = input_tokens.index(video_token_id, start)
                else:
                    end_video = len(input_tokens) + 1
                if end_image < end_video:
                    t, h, w = image_grid_thw[image_index]
                    image_index += 1
                    remain_images -= 1
                    end = end_image
                else:
                    t, h, w = video_grid_thw[video_index]
                    video_index += 1
                    remain_videos -= 1
                    end = end_video

                llm_grid_t = t.item()
                llm_grid_h = h.item() // spatial_merge_size
                llm_grid_w = w.item() // spatial_merge_size
                text_len = end - start

                text_len -= skip_vision_start_token[image_index - 1]
                text_len = max(0, text_len)

                start_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + start_idx)

                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()

                if skip_vision_start_token[image_index - 1]:
                    if fix_point > 0:
                        fix_point = fix_point - start_idx
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + fix_point + start_idx)
                    fix_point = 0
                else:
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + start_idx)
                start = end + llm_grid_t * llm_grid_h * llm_grid_w

            if start < len(input_tokens):
                start_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - start
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + start_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
    else:
        position_ids = (
            torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
        )
        mrope_position_deltas = torch.zeros(
            [input_ids.shape[0], 1],
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
    return position_ids, mrope_position_deltas


def _retrieve_timesteps(
    scheduler,
    num_inference_steps: int,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    shift: float | None = None,
):
    if shift is not None:
        if hasattr(scheduler, "set_shift"):
            scheduler.set_shift(shift)
        elif hasattr(scheduler.config, "flow_shift"):
            scheduler.register_to_config(flow_shift=shift)
        elif hasattr(scheduler.config, "shift"):
            scheduler.register_to_config(shift=shift)

    if timesteps is not None:
        scheduler.set_timesteps(num_inference_steps, device=device)
        scheduler.timesteps = torch.tensor(timesteps, device=device, dtype=torch.float32)
        sigmas = [float(t) / scheduler.config.num_train_timesteps for t in scheduler.timesteps]
        sigmas.append(0.0)
        scheduler.sigmas = torch.tensor(sigmas, device=device, dtype=torch.float32)
        if hasattr(scheduler, "num_inference_steps"):
            scheduler.num_inference_steps = len(timesteps)
        return scheduler.timesteps, len(timesteps)

    signature = inspect.signature(scheduler.set_timesteps)
    if "shift" in signature.parameters and shift is not None:
        scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device)
    return scheduler.timesteps, num_inference_steps


class HiDreamO1ImagePipeline(DiffusionPipeline):
    r"""
    Pipeline for HiDream-O1-Image.

    HiDream-O1 differs from latent diffusion pipelines because it does not use a VAE. The model denoises raw RGB pixel
    patches directly in a unified Qwen3-VL token space.

    Args:
        transformer (`PreTrainedModel`):
            HiDream-O1 compatible Qwen3-VL model. Its forward pass must accept `vinputs`, `timestep`, and `token_types`
            and return an object with an `x_pred` tensor.
        processor (`ProcessorMixin`):
            Processor used to apply the chat template and prepare reference images.
        scheduler (`FlowMatchEulerDiscreteScheduler` or `UniPCMultistepScheduler`):
            Scheduler used to denoise raw pixel patches.
    """

    model_cpu_offload_seq = "transformer"
    _callback_tensor_inputs = ["image_patches"]

    def __init__(
        self,
        transformer: PreTrainedModel,
        processor: ProcessorMixin,
        scheduler: FlowMatchEulerDiscreteScheduler | UniPCMultistepScheduler,
    ):
        super().__init__()

        self.register_modules(transformer=transformer, processor=processor, scheduler=scheduler)
        self.patch_size = DEFAULT_PATCH_SIZE
        self.condition_image_size = DEFAULT_CONDITION_IMAGE_SIZE
        self.default_sample_size = 2048

        tokenizer = _get_tokenizer(self.processor)
        _add_special_token_shortcuts(tokenizer)

    def _load_ref_images(self, ref_images: PipelineImageInput | list[PipelineImageInput] | None) -> list[PIL.Image.Image]:
        if ref_images is None:
            return []
        if isinstance(ref_images, (str, PIL.Image.Image, np.ndarray, torch.Tensor)):
            ref_images = [ref_images]

        loaded_images = []
        for image in ref_images:
            if isinstance(image, (str, PIL.Image.Image)):
                loaded_images.append(load_image(image))
            elif isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = np.clip(image * 255 if image.max() <= 1 else image, 0, 255).astype(np.uint8)
                loaded_images.append(Image.fromarray(image).convert("RGB"))
            elif isinstance(image, torch.Tensor):
                image = image.detach().cpu()
                if image.ndim == 3 and image.shape[0] in (1, 3):
                    image = image.permute(1, 2, 0)
                image = image.float().numpy()
                image = np.clip(image * 255 if image.max() <= 1 else image, 0, 255).astype(np.uint8)
                loaded_images.append(Image.fromarray(image).convert("RGB"))
            else:
                raise ValueError(
                    "`ref_images` must be a PIL image, path, URL, numpy array, torch tensor, or a list of those."
                )
        return loaded_images

    def _patchify_pixels(self, image: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = image.shape
        patch_size = self.patch_size
        image = image.reshape(
            batch_size,
            channels,
            height // patch_size,
            patch_size,
            width // patch_size,
            patch_size,
        )
        image = image.permute(0, 2, 4, 1, 3, 5)
        return image.reshape(batch_size, (height // patch_size) * (width // patch_size), channels * patch_size**2)

    def _unpatchify_pixels(self, patches: torch.Tensor, height: int, width: int) -> torch.Tensor:
        patch_size = self.patch_size
        batch_size = patches.shape[0]
        patches = patches.reshape(
            batch_size,
            height // patch_size,
            width // patch_size,
            3,
            patch_size,
            patch_size,
        )
        patches = patches.permute(0, 3, 1, 4, 2, 5)
        return patches.reshape(batch_size, 3, height, width)

    def _pil_to_patches(self, image: PIL.Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        array = np.asarray(image.convert("RGB")).astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return self._patchify_pixels(tensor).squeeze(0).to(device=device, dtype=dtype)

    def _build_t2i_text_sample(
        self,
        prompt: str,
        height: int,
        width: int,
        tokenizer,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        model_config = self.transformer.config
        image_token_id = model_config.image_token_id
        video_token_id = model_config.video_token_id
        vision_start_token_id = model_config.vision_start_token_id
        image_len = (height // self.patch_size) * (width // self.patch_size)

        messages = [{"role": "user", "content": prompt}]
        template_caption = (
            self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            + tokenizer.boi_token
            + tokenizer.tms_token * TIMESTEP_TOKEN_NUM
        )
        input_ids = tokenizer.encode(template_caption, return_tensors="pt", add_special_tokens=False)

        image_grid_thw = torch.tensor([1, height // self.patch_size, width // self.patch_size], dtype=torch.int64)
        image_grid_thw = image_grid_thw.unsqueeze(0)

        vision_tokens = torch.full((1, image_len), image_token_id, dtype=input_ids.dtype)
        vision_tokens[0, 0] = vision_start_token_id
        input_ids_pad = torch.cat([input_ids, vision_tokens], dim=-1)

        position_ids, _ = _get_rope_index_fix_point(
            1,
            image_token_id,
            video_token_id,
            vision_start_token_id,
            input_ids=input_ids_pad,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=None,
            skip_vision_start_token=[1],
        )

        text_seq_len = input_ids.shape[-1]
        all_seq_len = position_ids.shape[-1]

        token_types = torch.zeros((1, all_seq_len), dtype=input_ids.dtype)
        begin = text_seq_len - TIMESTEP_TOKEN_NUM
        token_types[0, begin : begin + image_len + TIMESTEP_TOKEN_NUM] = 1
        token_types[0, text_seq_len - TIMESTEP_TOKEN_NUM : text_seq_len] = 3

        vinput_mask = token_types == 1
        token_types_bin = (token_types > 0).to(token_types.dtype)

        sample = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "token_types": token_types_bin,
            "vinput_mask": vinput_mask,
        }
        return {key: value.to(device) for key, value in sample.items()}

    def _build_reference_samples(
        self,
        prompt: str,
        ref_images: list[PIL.Image.Image],
        height: int,
        width: int,
        max_size: int,
        tokenizer,
        device: torch.device,
        dtype: torch.dtype,
        guidance_scale: float,
        layout_bboxes: str | list | dict | None,
        preresized_ref_image: PIL.Image.Image | None,
    ):
        model_config = self.transformer.config
        image_token_id = model_config.image_token_id
        video_token_id = model_config.video_token_id
        vision_start_token_id = model_config.vision_start_token_id
        spatial_merge_size = getattr(model_config.vision_config, "spatial_merge_size", 1)

        layout_data = None
        num_ref_images = len(ref_images)
        if layout_bboxes is not None and preresized_ref_image is None:
            layout_data = _load_layout_bboxes(layout_bboxes)
            num_ref_images += 1

        if layout_data is not None:
            ref_images = _create_layout_reference_images(
                ref_images=ref_images,
                layout_bboxes=layout_data,
                image_width=width,
                image_height=height,
                ref_max_size=max_size,
                patch_size=self.patch_size,
            )

        ref_images_resized = []
        ref_patches = []
        for image in ref_images:
            if preresized_ref_image is not None and image is preresized_ref_image:
                resized_image = image
            else:
                resized_image = _resize_pil_image(image, max_size, self.patch_size)
            ref_images_resized.append(resized_image)
            ref_patches.append(self._pil_to_patches(resized_image, device=device, dtype=dtype))

        ref_image_lens = [patches.shape[0] for patches in ref_patches]
        total_ref_len = sum(ref_image_lens)
        ref_patches = torch.cat(ref_patches, dim=0).unsqueeze(0).to(device=device, dtype=dtype)
        target_image_len = (height // self.patch_size) * (width // self.patch_size)

        if num_ref_images <= 4:
            cond_img_size = self.condition_image_size
        elif num_ref_images <= 8:
            cond_img_size = self.condition_image_size * 48 // 64
        else:
            cond_img_size = self.condition_image_size // 2

        ref_images_vlm = []
        for resized_image in ref_images_resized:
            cond_w, cond_h = _calculate_dimensions(cond_img_size, resized_image.width / resized_image.height)
            ref_images_vlm.append(resized_image.resize((cond_w, cond_h), resample=Image.LANCZOS))

        image_grid_thw_tgt = torch.tensor([1, height // self.patch_size, width // self.patch_size], dtype=torch.int64)
        image_grid_thw_tgt = image_grid_thw_tgt.unsqueeze(0)
        image_grid_thw_ref = torch.zeros((num_ref_images, 3), dtype=torch.int64)
        for i, resized_image in enumerate(ref_images_resized):
            rw, rh = resized_image.size
            image_grid_thw_ref[i] = torch.tensor([1, rh // self.patch_size, rw // self.patch_size], dtype=torch.int64)

        samples = []
        captions = [prompt]
        if guidance_scale > 1.0:
            captions.append(" ")

        for caption in captions:
            content = [{"type": "image"} for _ in range(num_ref_images)]
            content.append({"type": "text", "text": caption})
            messages = [{"role": "user", "content": content}]
            template_caption = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            proc = self.processor(text=[template_caption], images=ref_images_vlm, padding="longest", return_tensors="pt")
            input_ids_2 = tokenizer.encode(
                tokenizer.boi_token + tokenizer.tms_token * TIMESTEP_TOKEN_NUM,
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_ids = torch.cat([proc.input_ids, input_ids_2], dim=-1)

            image_grid_thw_cond = proc.image_grid_thw.clone()
            for i in range(num_ref_images):
                image_grid_thw_cond[i, 1] //= spatial_merge_size
                image_grid_thw_cond[i, 2] //= spatial_merge_size
            image_grid_thw_all = torch.cat([image_grid_thw_cond, image_grid_thw_tgt, image_grid_thw_ref], dim=0)

            vision_tokens_list = []
            vision_tokens_tgt = torch.full((1, target_image_len), image_token_id, dtype=input_ids.dtype)
            vision_tokens_tgt[0, 0] = vision_start_token_id
            vision_tokens_list.append(vision_tokens_tgt)
            for ref_len in ref_image_lens:
                vision_tokens_ref = torch.full((1, ref_len), image_token_id, dtype=input_ids.dtype)
                vision_tokens_ref[0, 0] = vision_start_token_id
                vision_tokens_list.append(vision_tokens_ref)
            vision_tokens = torch.cat(vision_tokens_list, dim=1)
            input_ids_pad = torch.cat([input_ids, vision_tokens], dim=-1)

            position_ids, _ = _get_rope_index_fix_point(
                1,
                image_token_id,
                video_token_id,
                vision_start_token_id,
                input_ids=input_ids_pad,
                image_grid_thw=image_grid_thw_all,
                video_grid_thw=None,
                attention_mask=None,
                skip_vision_start_token=[0] * num_ref_images + [1] + [1] * num_ref_images,
            )

            text_seq_len = input_ids.shape[-1]
            all_seq_len = position_ids.shape[-1]

            token_types_raw = torch.zeros((1, all_seq_len), dtype=input_ids.dtype)
            begin = text_seq_len - TIMESTEP_TOKEN_NUM
            end = begin + target_image_len + TIMESTEP_TOKEN_NUM
            token_types_raw[0, begin:end] = 1
            token_types_raw[0, end : end + total_ref_len] = 2
            token_types_raw[0, text_seq_len - TIMESTEP_TOKEN_NUM : text_seq_len] = 3

            vinput_mask = torch.logical_or(token_types_raw == 1, token_types_raw == 2)
            token_types_bin = (token_types_raw > 0).to(token_types_raw.dtype)

            samples.append(
                {
                    "input_ids": input_ids.to(device),
                    "position_ids": position_ids.to(device),
                    "token_types": token_types_bin.to(device),
                    "vinput_mask": vinput_mask.to(device),
                    "pixel_values": proc.pixel_values.to(device, dtype),
                    "image_grid_thw": proc.image_grid_thw.to(device),
                }
            )

        return samples, ref_patches, target_image_len

    def _decode_patches(self, patches: torch.Tensor, height: int, width: int, output_type: str):
        image = (self._unpatchify_pixels(patches.float(), height, width) + 1.0) / 2.0
        image = image.clamp(0.0, 1.0)

        if output_type == "pt":
            return image

        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "np":
            return image
        if output_type == "pil":
            return numpy_to_pil(image)
        raise ValueError("`output_type` must be one of 'pil', 'np', 'pt', or 'latent'.")

    def check_inputs(
        self,
        prompt,
        height: int,
        width: int,
        callback_on_step_end_tensor_inputs: list[str],
    ):
        if prompt is None:
            raise ValueError("`prompt` must be provided.")
        if isinstance(prompt, list) and len(prompt) != 1:
            raise ValueError("HiDream-O1 pipeline currently supports a single prompt per call.")
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(f"`height` and `width` must be divisible by {self.patch_size}.")
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found "
                f"{[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def interrupt(self):
        return self._interrupt

    def _get_execution_device(self):
        try:
            return self._execution_device
        except AttributeError:
            pass

        try:
            return self.device
        except AttributeError:
            pass

        if isinstance(self.transformer, torch.nn.Module):
            try:
                return next(self.transformer.parameters()).device
            except StopIteration:
                pass
            try:
                return next(self.transformer.buffers()).device
            except StopIteration:
                pass

        return torch.device("cpu")

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | list[str],
        ref_images: PipelineImageInput | list[PipelineImageInput] | None = None,
        height: int = 2048,
        width: int = 2048,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        image_patches: torch.Tensor | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        model_type: str = "full",
        scheduler_type: str | None = None,
        timesteps: list[int] | None = None,
        shift: float | None = None,
        noise_scale_start: float = DEFAULT_NOISE_SCALE,
        noise_scale_end: float = DEFAULT_NOISE_SCALE,
        noise_clip_std: float = 0.0,
        keep_original_aspect: bool = False,
        layout_bboxes: str | list | dict | None = None,
        use_resolution_binning: bool = True,
        use_flash_attn: bool = True,
        callback_on_step_end: Callable[[int, int], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["image_patches"],
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `list[str]`):
                Prompt to guide image generation. The first implementation supports one prompt per call.
            ref_images (`PipelineImageInput` or `list[PipelineImageInput]`, *optional*):
                Reference image or images for editing or subject-driven generation.
            height (`int`, defaults to `2048`):
                Height in pixels. The final value must be divisible by `32`.
            width (`int`, defaults to `2048`):
                Width in pixels. The final value must be divisible by `32`.
            num_inference_steps (`int`, *optional*):
                Number of denoising steps. Defaults to `50` for `model_type="full"` and `28` for `model_type="dev"`.
            guidance_scale (`float`, *optional*):
                Classifier-free guidance scale. Defaults to `5.0` for full and `0.0` for dev.
            generator (`torch.Generator`, *optional*):
                Random generator for deterministic sampling.
            image_patches (`torch.Tensor`, *optional*):
                Pre-generated noisy pixel patches of shape `(1, height / 32 * width / 32, 3 * 32 * 32)`.
            output_type (`str`, defaults to `"pil"`):
                Output type. One of `"pil"`, `"np"`, `"pt"`, or `"latent"`.
            model_type (`str`, defaults to `"full"`):
                Preset selector matching the official scripts. Use `"full"` or `"dev"`.
            scheduler_type (`str`, *optional*):
                One of `"default"`, `"flow_match"`, or `"flash"`. Defaults follow the official script.
            timesteps (`list[int]`, *optional*):
                Custom timestep list. Dev defaults to the official 28-step list.
            shift (`float`, *optional*):
                Flow timestep shift. Defaults to `3.0` for full and `1.0` for dev.
            keep_original_aspect (`bool`, defaults to `False`):
                With exactly one reference image, resize it to a patch-aligned max size and use its aspect ratio.
            layout_bboxes (`str`, `list`, or `dict`, *optional*):
                Layout boxes in xxyy relative coordinates.
            use_resolution_binning (`bool`, defaults to `True`):
                Snap `height` and `width` to one of the official predefined resolutions.
            use_flash_attn (`bool`, defaults to `True`):
                Passed to compatible HiDream-O1 transformer implementations.
            callback_on_step_end (`Callable`, *optional*):
                Function called at the end of each denoising step.

        Returns:
            [`~pipelines.hidream_o1.HiDreamO1ImagePipelineOutput`] or `tuple`.

        Examples:
        """
        if model_type not in {"full", "dev"}:
            raise ValueError("`model_type` must be either 'full' or 'dev'.")
        if scheduler_type is not None and scheduler_type not in {"default", "flow_match", "flash"}:
            raise ValueError("`scheduler_type` must be one of 'default', 'flow_match', or 'flash'.")

        ref_images = self._load_ref_images(ref_images)
        is_editing = len(ref_images) == 1

        if num_inference_steps is None:
            num_inference_steps = 50 if model_type == "full" else 28
        if guidance_scale is None:
            guidance_scale = 5.0 if model_type == "full" else 0.0
        if shift is None:
            shift = 3.0 if model_type == "full" else 1.0
        if scheduler_type is None:
            scheduler_type = "default" if model_type == "full" else "flow_match" if is_editing else "flash"
        if timesteps is None and model_type == "dev":
            timesteps = DEFAULT_TIMESTEPS

        preresized_ref_image = None
        if keep_original_aspect and len(ref_images) == 1:
            preresized_ref_image = _resize_pil_image(ref_images[0], 2048, self.patch_size)
            ref_images = [preresized_ref_image]
            width, height = preresized_ref_image.size
        elif use_resolution_binning:
            binned_width, binned_height = _find_closest_resolution(width, height)
            if binned_width != width or binned_height != height:
                logger.warning(f"Resolution snapped from {width}x{height} to {binned_width}x{binned_height}.")
            width, height = binned_width, binned_height

        self.check_inputs(prompt, height, width, callback_on_step_end_tensor_inputs)
        prompt = prompt[0] if isinstance(prompt, list) else prompt
        self._guidance_scale = guidance_scale
        self._interrupt = False

        device = self._get_execution_device()
        dtype = getattr(self.transformer, "dtype", torch.float32)
        if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            dtype = torch.float32

        tokenizer = _get_tokenizer(self.processor)
        _add_special_token_shortcuts(tokenizer)

        if not ref_images:
            cond_sample = self._build_t2i_text_sample(prompt, height, width, tokenizer, device)
            samples = [cond_sample]
            if guidance_scale > 1.0:
                samples.append(self._build_t2i_text_sample(" ", height, width, tokenizer, device))
            ref_patches = None
            target_image_len = (height // self.patch_size) * (width // self.patch_size)
        else:
            num_ref_images = len(ref_images) + (1 if layout_bboxes is not None and preresized_ref_image is None else 0)
            if num_ref_images == 1:
                max_size = max(height, width)
            elif num_ref_images == 2:
                max_size = max(height, width) * 48 // 64
            elif num_ref_images <= 4:
                max_size = max(height, width) // 2
            elif num_ref_images <= 8:
                max_size = max(height, width) * 24 // 64
            else:
                max_size = max(height, width) // 4

            samples, ref_patches, target_image_len = self._build_reference_samples(
                prompt=prompt,
                ref_images=ref_images,
                height=height,
                width=width,
                max_size=max_size,
                tokenizer=tokenizer,
                device=device,
                dtype=dtype,
                guidance_scale=guidance_scale,
                layout_bboxes=layout_bboxes,
                preresized_ref_image=preresized_ref_image,
            )

        noise_shape = (1, 3, height, width)
        if image_patches is None:
            noise = noise_scale_start * randn_tensor(noise_shape, generator=generator, device=device, dtype=dtype)
            image_patches = self._patchify_pixels(noise)
        else:
            expected_shape = (1, (height // self.patch_size) * (width // self.patch_size), 3 * self.patch_size**2)
            if tuple(image_patches.shape) != expected_shape:
                raise ValueError(f"Unexpected `image_patches` shape {image_patches.shape}, expected {expected_shape}.")
            image_patches = image_patches.to(device=device, dtype=dtype)

        if XLA_AVAILABLE:
            timestep_device = "cpu"
        else:
            timestep_device = device
        timesteps_tensor, num_inference_steps = _retrieve_timesteps(
            self.scheduler,
            num_inference_steps=num_inference_steps,
            device=timestep_device,
            timesteps=timesteps,
            shift=shift,
        )

        if len(timesteps_tensor) > 1:
            noise_scale_schedule = [
                noise_scale_start + (noise_scale_end - noise_scale_start) * i / (len(timesteps_tensor) - 1)
                for i in range(len(timesteps_tensor))
            ]
        else:
            noise_scale_schedule = [noise_scale_start]

        cond_image_embeds = None
        cond_deepstack_image_embeds = None

        def forward_once(
            sample,
            patches_in,
            timestep_pixeldit,
            precomputed_image_embeds=None,
            precomputed_deepstack_image_embeds=None,
        ):
            kwargs = {
                "input_ids": sample["input_ids"],
                "position_ids": sample["position_ids"],
                "vinputs": patches_in,
                "timestep": timestep_pixeldit.reshape(-1).to(device),
                "token_types": sample["token_types"],
                "use_flash_attn": use_flash_attn,
                "precomputed_image_embeds": precomputed_image_embeds,
                "precomputed_deepstack_image_embeds": precomputed_deepstack_image_embeds,
            }
            if "pixel_values" in sample:
                kwargs["pixel_values"] = sample["pixel_values"]
            if "image_grid_thw" in sample:
                kwargs["image_grid_thw"] = sample["image_grid_thw"]

            outputs = self.transformer(**kwargs)
            x_pred = outputs.x_pred
            image_embeds = getattr(outputs, "cond_image_embeds", None)
            deepstack_image_embeds = getattr(outputs, "cond_deepstack_image_embeds", None)

            if ref_patches is None:
                return x_pred[0, sample["vinput_mask"][0]].unsqueeze(0)
            return x_pred[0, sample["vinput_mask"][0]][:target_image_len].unsqueeze(0), image_embeds, deepstack_image_embeds

        with self.progress_bar(total=len(timesteps_tensor)) as progress_bar:
            for step_idx, step_t in enumerate(timesteps_tensor):
                if self.interrupt:
                    continue

                step_t = step_t.to(device)
                timestep_pixeldit = 1.0 - step_t.float() / self.scheduler.config.num_train_timesteps
                sigma = (step_t.float() / self.scheduler.config.num_train_timesteps).to(dtype=torch.float32)
                sigma = sigma.clamp_min(T_EPS)

                if ref_patches is None:
                    x_pred_cond = forward_once(samples[0], image_patches.clone(), timestep_pixeldit)
                    v_cond = (x_pred_cond.to(dtype=torch.float32) - image_patches.to(dtype=torch.float32)) / sigma

                    if len(samples) > 1:
                        x_pred_uncond = forward_once(samples[1], image_patches.clone(), timestep_pixeldit)
                        v_uncond = (x_pred_uncond.to(dtype=torch.float32) - image_patches.to(dtype=torch.float32)) / sigma
                        v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
                    else:
                        v_guided = v_cond
                else:
                    model_inputs = torch.cat([image_patches, ref_patches], dim=1)
                    x_vis_list = []
                    for sample in samples:
                        x_pred, image_embeds, deepstack_image_embeds = forward_once(
                            sample,
                            model_inputs,
                            timestep_pixeldit,
                            precomputed_image_embeds=cond_image_embeds,
                            precomputed_deepstack_image_embeds=cond_deepstack_image_embeds,
                        )
                        if image_embeds is not None and deepstack_image_embeds is not None:
                            cond_image_embeds = image_embeds.detach()
                            cond_deepstack_image_embeds = [emb.detach() for emb in deepstack_image_embeds]
                        x_vis_list.append(x_pred)

                    x_vis_stacked = torch.cat(x_vis_list, dim=0)
                    patches_repeated = image_patches.expand(len(samples), -1, -1)
                    v_pred = (x_vis_stacked.to(dtype=torch.float32) - patches_repeated.to(dtype=torch.float32)) / sigma

                    v_cond = v_pred[0:1]
                    if len(samples) > 1:
                        v_uncond = v_pred[1:]
                        v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
                    else:
                        v_guided = v_cond

                model_output = -v_guided
                if scheduler_type == "flash":
                    sample = image_patches.to(torch.float32)
                    scheduler_step_index = getattr(self.scheduler, "step_index", None)
                    if scheduler_step_index is None:
                        scheduler_step_index = step_idx
                    sigma_step = self.scheduler.sigmas[scheduler_step_index].to(device=device)
                    denoised = sample - model_output.float() * sigma_step
                    sigma_next = self.scheduler.sigmas[scheduler_step_index + 1].to(device=device)
                    noise = randn_tensor(
                        model_output.shape,
                        generator=generator,
                        device=model_output.device,
                        dtype=denoised.dtype,
                    )
                    if noise_clip_std > 0:
                        clip_val = noise_clip_std * noise.std().item()
                        noise = noise.clamp(min=-clip_val, max=clip_val)
                    image_patches = (
                        sigma_next * noise * noise_scale_schedule[step_idx] + (1.0 - sigma_next) * denoised
                    ).to(dtype)
                    if getattr(self.scheduler, "_step_index", None) is None:
                        self.scheduler._step_index = 0
                    self.scheduler._step_index += 1
                else:
                    patches_dtype = image_patches.dtype
                    image_patches = self.scheduler.step(
                        model_output.float(),
                        step_t.to(dtype=torch.float32),
                        image_patches.float(),
                        return_dict=False,
                    )[0].to(patches_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for key in callback_on_step_end_tensor_inputs:
                        callback_kwargs[key] = locals()[key]
                    callback_outputs = callback_on_step_end(self, step_idx, step_t, callback_kwargs)
                    image_patches = callback_outputs.pop("image_patches", image_patches)

                progress_bar.update()
                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = image_patches
        else:
            image = self._decode_patches(image_patches, height, width, output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        return HiDreamO1ImagePipelineOutput(images=image)
