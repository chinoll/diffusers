<!-- Copyright 2026 chinoll and The HuggingFace Team. All rights reserved.
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
# limitations under the License. -->

# HiDream-O1

[HiDream-O1-Image](https://huggingface.co/HiDream-ai/HiDream-O1-Image) is a raw pixel patch diffusion model.
Unlike [`HiDreamImagePipeline`], [`HiDreamO1ImagePipeline`] does not use a VAE. It denoises RGB image patches directly with a compatible Qwen3-VL transformer.

## Available models

The following model is available for the [`HiDreamO1ImagePipeline`] pipeline:

| Model name | Description |
|:---|:---|
| [`HiDream-ai/HiDream-O1-Image`](https://huggingface.co/HiDream-ai/HiDream-O1-Image) | Raw pixel patch image generation and editing model. |

## HiDreamO1ImagePipeline

[[autodoc]] HiDreamO1ImagePipeline
  - all
  - __call__

## HiDreamO1ImagePipelineOutput

[[autodoc]] pipelines.hidream_o1.pipeline_output.HiDreamO1ImagePipelineOutput
