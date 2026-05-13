# coding=utf-8
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

import types
import unittest

import numpy as np
import torch

from diffusers import FlowMatchEulerDiscreteScheduler, HiDreamO1ImagePipeline

from ...testing_utils import enable_full_determinism


enable_full_determinism()


class DummyTokenizer:
    boi_token = "<|boi_token|>"
    bor_token = "<|bor_token|>"
    eor_token = "<|eor_token|>"
    bot_token = "<|bot_token|>"
    tms_token = "<|tms_token|>"

    def encode(self, text, return_tensors=None, add_special_tokens=False):
        ids = [11, 12]
        if self.tms_token in text:
            ids.append(151673)
        tensor = torch.tensor([ids], dtype=torch.long)
        if return_tensors == "pt":
            return tensor
        return ids


class DummyProcessor:
    def __init__(self):
        self.tokenizer = DummyTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        content = messages[0]["content"]
        if isinstance(content, list):
            text = " ".join(item.get("text", "") for item in content if item.get("type") == "text")
        else:
            text = content
        return f"<|user|>{text}<|assistant|>"


class DummyHiDreamO1Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))
        self.config = types.SimpleNamespace(
            image_token_id=100,
            video_token_id=101,
            vision_start_token_id=102,
            vision_config=types.SimpleNamespace(spatial_merge_size=1),
        )

    @property
    def dtype(self):
        return self.dummy.dtype

    def forward(
        self,
        input_ids,
        position_ids,
        vinputs,
        timestep,
        token_types,
        use_flash_attn=True,
        **kwargs,
    ):
        batch_size, seq_len = token_types.shape
        patch_dim = vinputs.shape[-1]
        x_pred = torch.zeros(batch_size, seq_len, patch_dim, device=vinputs.device, dtype=vinputs.dtype)
        return types.SimpleNamespace(x_pred=x_pred)


class HiDreamO1ImagePipelineFastTests(unittest.TestCase):
    def get_dummy_pipeline(self):
        transformer = DummyHiDreamO1Transformer()
        processor = DummyProcessor()
        scheduler = FlowMatchEulerDiscreteScheduler()
        pipe = HiDreamO1ImagePipeline(transformer=transformer, processor=processor, scheduler=scheduler)
        pipe.set_progress_bar_config(disable=None)
        return pipe

    def test_text_to_image_raw_pixel_patches(self):
        pipe = self.get_dummy_pipeline()
        generator = torch.Generator(device="cpu").manual_seed(0)

        output = pipe(
            "A tiny test image",
            height=64,
            width=64,
            num_inference_steps=1,
            guidance_scale=0.0,
            generator=generator,
            scheduler_type="flow_match",
            use_resolution_binning=False,
            output_type="np",
        )

        image = output.images
        self.assertEqual(image.shape, (1, 64, 64, 3))
        self.assertTrue(np.allclose(image, 0.5, atol=1e-5))

    def test_latent_output_shape(self):
        pipe = self.get_dummy_pipeline()
        generator = torch.Generator(device="cpu").manual_seed(0)

        output = pipe(
            "A tiny test image",
            height=64,
            width=96,
            num_inference_steps=1,
            guidance_scale=0.0,
            generator=generator,
            scheduler_type="flow_match",
            use_resolution_binning=False,
            output_type="latent",
        )

        self.assertEqual(output.images.shape, (1, 6, 3 * 32 * 32))

    def test_flash_scheduler_path(self):
        pipe = self.get_dummy_pipeline()
        generator = torch.Generator(device="cpu").manual_seed(0)

        output = pipe(
            "A tiny test image",
            height=64,
            width=64,
            num_inference_steps=1,
            guidance_scale=0.0,
            generator=generator,
            scheduler_type="flash",
            use_resolution_binning=False,
            output_type="np",
        )

        self.assertEqual(output.images.shape, (1, 64, 64, 3))
        self.assertTrue(np.allclose(output.images, 0.5, atol=1e-5))

    def test_multi_prompt_raises(self):
        pipe = self.get_dummy_pipeline()

        with self.assertRaisesRegex(ValueError, "single prompt"):
            pipe(
                ["first", "second"],
                height=64,
                width=64,
                num_inference_steps=1,
                scheduler_type="flow_match",
                use_resolution_binning=False,
            )


if __name__ == "__main__":
    unittest.main()
