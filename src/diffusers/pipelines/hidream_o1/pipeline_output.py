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

from dataclasses import dataclass

import numpy as np
import PIL.Image
import torch

from ...utils import BaseOutput


@dataclass
class HiDreamO1ImagePipelineOutput(BaseOutput):
    """
    Output class for HiDream-O1 image pipelines.

    Args:
        images (`list[PIL.Image.Image]`, `np.ndarray`, or `torch.Tensor`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`. Torch tensors are returned for `"pt"` images or `"latent"` raw pixel patches.
    """

    images: list[PIL.Image.Image] | np.ndarray | torch.Tensor
