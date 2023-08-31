# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""Data parser for Rodin dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import imageio
import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE

@dataclass
class RodinDataParserConfig(DataParserConfig):
    """Rodin dataset parser config"""

    _target: Type = field(default_factory=lambda: Rodin)
    """target class to instantiate"""
    data: Path = Path("/mnt/blob2/render_output_hd/_Aaron_Adams_5GHCX")
    """Directory specifying location of data."""
    scale_factor: float = 1.0 / 23.
    """How much to scale the camera origins by."""
    alpha_color: str = "white"
    """alpha color of background"""


@dataclass
class Rodin(DataParser):
    """Rodin Dataset
    Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.
    """

    config: RodinDataParserConfig

    def __init__(self, config: RodinDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color

    def _generate_dataparser_outputs(self, split="train"):
        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None

        # Rodin
        image_filenames = sorted([filename for filename in self.data.rglob("img_proc_fg_*.png")]) if split == 'train' else sorted([filename for filename in self.data.rglob("img_proc_fg_*.png")])[:5]
        CONSOLE.print(f"{self.data}_{split}: {len(image_filenames)}")
        poses = []
        for image_filename in image_filenames:
            image_id = image_filename.stem.split('_')[-1]
            meta = load_from_json(self.data / f"metadata_{image_id}.json")['cameras'][0]
            poses.append(np.array(meta['transformation']))
        poses = np.array(poses).astype(np.float32)
        
        image_width, image_height = int(meta['resolution'][0]), int(meta['resolution'][1])
        focal_length =  meta['focal_length'] / meta['sensor_width']  * image_width

        cx = image_width / 2.0
        cy = image_height / 2.0
        camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform

        # in x,y,z order
        camera_to_world[..., 3] *= self.scale_factor
        scene_box = SceneBox(aabb=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32))
        # scene_box = SceneBox(aabb=torch.tensor([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]], dtype=torch.float32)) # torch_ngp

        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
        )

        return dataparser_outputs