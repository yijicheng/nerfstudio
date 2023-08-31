# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

"""
Implementation of K-Planes (https://sarafridov.github.io/K-Planes/).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import nerfacc
import torch
from torch.nn import Parameter
from torchmetrics.image import PeakSignalNoiseRatio
# from torchmetrics.functional import structural_similarity_index_measure
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.losses import MSELoss, distortion_loss, interlevel_loss
from nerfstudio.model_components.ray_samplers import (
    VolumetricSampler,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, misc
from nerfstudio.utils.comms import get_world_size
from nerfstudio.utils.rich_utils import CONSOLE

from nerfstudio.fields.kplanes_field import KPlanesDensityField, KPlanesField


@dataclass
class KPlanesNGPModelConfig(ModelConfig):
    """K-Planes Model Config"""

    _target: Type = field(default_factory=lambda: KPlanesNGPModel)

    near_plane: float = 2.0
    """How far along the ray to start sampling."""

    far_plane: float = 6.0
    """How far along the ray to stop sampling."""

    grid_base_resolution: List[int] = field(default_factory=lambda: [128, 128, 128])
    """Base grid resolution."""

    grid_feature_dim: int = 32
    """Dimension of feature vectors stored in grid."""

    multiscale_res: List[int] = field(default_factory=lambda: [1, 2, 4])
    """Multiscale grid resolutions."""

    is_contracted: bool = False
    """Whether to use scene contraction (set to true for unbounded scenes)."""

    concat_features_across_scales: bool = True
    """Whether to concatenate features at different scales."""

    linear_decoder: bool = False
    """Whether to use a linear decoder instead of an MLP."""

    linear_decoder_layers: Optional[int] = 1
    """Number of layers in linear decoder"""

    sigma_decoder_layers: Optional[int] = 1
    """Number of layers in sigma decoder"""

    color_decoder_layers: Optional[int] = 2
    """Number of layers in color decoder"""

    sigma_decoder_hiddens: Optional[int] = 64
    """Number of hiddens in sigma decoder"""

    color_decoder_hiddens: Optional[int] = 64
    """Number of hiddens in color decoder"""

    appearance_embedding_dim: int = 0
    """Dimension of appearance embedding. Set to 0 to disable."""

    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""

    background_color: Literal["random", "last_sample", "black", "white"] = "white"
    """The background color as RGB."""

    loss_coefficients: Dict[str, float] = to_immutable_dict(
        {
            "img": 1.0,
            "plane_tv": 0.0001,
            "l1_time_planes": 0.0001,
            "time_smoothness": 0.1,
            "orientation_loss": 0.0001, # Orientation loss multiplier on computed normals.
            "pred_normal_loss": 0.001, # Predicted normal loss multiplier.
        }
    )
    """Loss coefficients."""

    subject_checkpoint: Optional[str] = None
    """Subject checkpoint."""

    decoder_checkpoint: Optional[str] = None
    """Decoder checkpoint."""

    freeze_decoder: bool = False
    """Whether to freeze decoder"""

    # occupancy grid sampling arguments
    grid_resolution: int = 128
    """Resolution of the grid used for the field."""

    grid_levels: int = 4
    """Levels of the grid used for the field."""

    alpha_thre: float = 0.01
    """Threshold for opacity skipping."""

    cone_angle: float = 0.004
    """Should be set to 0.0 for blender scenes but 1./256 for real scenes."""

    render_step_size: Optional[float] = None
    """Minimum step size for rendering."""

    render_step: Optional[int] = 1024
    """Maximum steps for rendering."""

    predict_normals: bool = False
    """Whether to predict normals or not."""

    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""


    multiple_fitting: bool = False


class KPlanesNGPModel(Model):
    config: KPlanesNGPModelConfig
    """K-Planes model

    Args:
        config: K-Planes configuration to instantiate model
    """
    def __init__(self, config: KPlanesNGPModelConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.is_contracted:
            scene_contraction = SceneContraction(order=float("inf"))
        else:
            scene_contraction = None

        # Fields
        self.field = KPlanesField(
            self.scene_box.aabb, # type: ignore
            num_images=self.num_train_data,
            grid_base_resolution=self.config.grid_base_resolution,
            grid_feature_dim=self.config.grid_feature_dim,
            concat_across_scales=self.config.concat_features_across_scales,
            multiscale_res=self.config.multiscale_res,
            spatial_distortion=scene_contraction,
            appearance_embedding_dim=self.config.appearance_embedding_dim,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            linear_decoder=self.config.linear_decoder,
            linear_decoder_layers=self.config.linear_decoder_layers,
        )

        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)

        if self.config.render_step_size is None:
            # auto step size: ~1024 samples in the base level grid
            self.config.render_step_size = ((self.scene_aabb[3:] - self.scene_aabb[:3]) ** 2).sum().sqrt().item() / self.config.render_step # type: ignore
        # Occupancy Grid.
        self.occupancy_grid = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            levels=self.config.grid_levels,
        )

        # Sampler
        self.sampler = VolumetricSampler(
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field.density_fn,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        # self.ssim = structural_similarity_index_measure
        # self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.temporal_distortion = len(self.config.grid_base_resolution) == 4  # for viewer

        if self.config.multiple_fitting and get_world_size() > 1:
            self._set_ddp_ignore()

    def _set_ddp_ignore(self):
        self._ddp_params_and_buffers_to_ignore = [
            item[0] for item in self.named_parameters() if not ('sigma_net' in item[0] or 'color_net' in item[0])
        ] + [
            item[0] for item in self.named_buffers()
        ]

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {
            "field.grids": list(self.field.grids.parameters()),
            "field.sigma_net": list(self.field.sigma_net.parameters()),
            "field.color_net": list(self.field.color_net.parameters()),
        }
        return param_groups
    
    def _reload_checkpoint(self) -> None:
        if self.config.subject_checkpoint is not None:
            CONSOLE.log(f"Loading subject checkpoint from {self.config.subject_checkpoint}")
            model_state_dict = torch.load(self.config.subject_checkpoint, map_location="cpu")['pipeline']
            is_ddp_model_state = get_world_size() > 1
            if not is_ddp_model_state:
                for key in model_state_dict.keys():
                    if "module" in key:
                        is_ddp_model_state = True

            grids_state_dict = dict()
            for key in self.field.grids.state_dict().keys():
                grids_state_dict[key] = model_state_dict[f"_model.module.field.grids.{key}"] if is_ddp_model_state else model_state_dict[f"_model.field.grids.{key}"]
            self.field.grids.load_state_dict(grids_state_dict)

            occupancy_grid_state_dict = dict()
            for key in self.occupancy_grid.state_dict().keys():
                occupancy_grid_state_dict[key] = model_state_dict[f"_model.module.occupancy_grid.{key}"] if is_ddp_model_state else model_state_dict[f"_model.occupancy_grid.{key}"]
            self.occupancy_grid.load_state_dict(occupancy_grid_state_dict)

            sampler_state_dict = dict()
            for key in self.sampler.state_dict().keys():
                sampler_state_dict[key] = model_state_dict[f"_model.module.sampler.{key}"] if is_ddp_model_state else model_state_dict[f"_model.sampler.{key}"]
            self.sampler.load_state_dict(sampler_state_dict)

            del model_state_dict, grids_state_dict, occupancy_grid_state_dict, 
        else:
            CONSOLE.print("No subject checkpoint to load, so training from scratch.")


        if self.config.decoder_checkpoint is not None:
            CONSOLE.log(f"Loading decoder checkpoint from {self.config.decoder_checkpoint}")
            model_state_dict = torch.load(self.config.decoder_checkpoint, map_location="cpu")['pipeline']
            is_ddp_model_state = get_world_size() > 1
            if not is_ddp_model_state:
                for key in model_state_dict.keys():
                    if "module" in key:
                        is_ddp_model_state = True
                        break

            sigma_net_state_dict = dict()
            for key in self.field.sigma_net.state_dict().keys():
                sigma_net_state_dict[key] =  model_state_dict[f"_model.module.field.sigma_net.{key}"] if is_ddp_model_state else model_state_dict[f"_model.field.sigma_net.{key}"]
            self.field.sigma_net.load_state_dict(sigma_net_state_dict)

            color_net_state_dict = dict()
            for key in self.field.color_net.state_dict().keys():
                color_net_state_dict[key] =  model_state_dict[f"_model.module.field.color_net.{key}"] if is_ddp_model_state else model_state_dict[f"_model.field.color_net.{key}"]
            self.field.color_net.load_state_dict(color_net_state_dict)
            del model_state_dict, sigma_net_state_dict, color_net_state_dict
        else:
            CONSOLE.print("No decoder checkpoint to load, so training from scratch.")

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        def update_occupancy_grid(step: int):
            self.occupancy_grid.update_every_n_steps(
                step=step,
                occ_eval_fn=lambda x: self.field.density_fn(x) * self.config.render_step_size,
            )

        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=update_occupancy_grid,
            ),
        ]

    def get_outputs(self, ray_bundle: RayBundle):
        num_rays = len(ray_bundle)

        with torch.no_grad():
            ray_samples, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
                near_plane=self.config.near_plane,
                far_plane=self.config.far_plane,
                render_step_size=self.config.render_step_size,
                alpha_thre=self.config.alpha_thre,
                cone_angle=self.config.cone_angle,
            )

        field_outputs = self.field(ray_samples)

        # accumulation
        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        weights = nerfacc.render_weight_from_density(
            t_starts=ray_samples.frustums.starts[..., 0],
            t_ends=ray_samples.frustums.ends[..., 0],
            sigmas=field_outputs[FieldHeadNames.DENSITY][..., 0],
            packed_info=packed_info,
        )[0]
        weights = weights[..., None]

        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )
        depth = self.renderer_depth(
            weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays
        )
        accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "num_samples_per_ray": packed_info[:, 1],
        }

        return outputs

    def get_metrics_dict(self, outputs, batch):
        image = batch["image"].to(self.device)

        metrics_dict = {
            "psnr": self.psnr(outputs["rgb"], image)
        }
        if self.training:
            field_grids = [g.plane_coefs for g in self.field.grids]

            metrics_dict["plane_tv"] = space_tv_loss(field_grids) # type: ignore

            if len(self.config.grid_base_resolution) == 4:
                metrics_dict["l1_time_planes"] = l1_time_planes(field_grids) # type: ignore
                metrics_dict["time_smoothness"] = time_smoothness(field_grids) # type: ignore

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"].to(self.device)

        loss_dict = {"rgb": self.rgb_loss(image, outputs["rgb"])}
        if self.training:
            for key in self.config.loss_coefficients:
                if key in metrics_dict: # type: ignore
                    loss_dict[key] = metrics_dict[key].clone() # type: ignore

            loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)

        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(outputs["depth"], accumulation=outputs["accumulation"])

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        # all of these metrics will be logged as scalars
        metrics_dict = {
            "psnr": float(self.psnr(image, rgb).item()),
            # "ssim": float(self.ssim(image, rgb)), # type: ignore
            # "lpips": float(self.lpips(image, rgb))
        }
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        return metrics_dict, images_dict


def compute_plane_tv(t: torch.Tensor, only_w: bool = False) -> float:
    """Computes total variance across a plane.

    Args:
        t: Plane tensor
        only_w: Whether to only compute total variance across w dimension

    Returns:
        Total variance
    """
    _, h, w = t.shape
    w_tv = torch.square(t[..., :, 1:] - t[..., :, : w - 1]).mean()

    if only_w:
        return w_tv # type: ignore

    h_tv = torch.square(t[..., 1:, :] - t[..., : h - 1, :]).mean()
    return h_tv + w_tv # type: ignore


def space_tv_loss(multi_res_grids: List[torch.Tensor]) -> float:
    """Computes total variance across each spatial plane in the grids.

    Args:
        multi_res_grids: Grids to compute total variance over

    Returns:
        Total variance
    """

    total = 0.0
    num_planes = 0
    for grids in multi_res_grids:
        if len(grids) == 3:
            spatial_planes = {0, 1, 2}
        else:
            spatial_planes = {0, 1, 3}

        for grid_id, grid in enumerate(grids):
            if grid_id in spatial_planes:
                total += compute_plane_tv(grid)
            else:
                # Space is the last dimension for space-time planes.
                total += compute_plane_tv(grid, only_w=True)
            num_planes += 1
    return total / num_planes


def l1_time_planes(multi_res_grids: List[torch.Tensor]) -> float:
    """Computes the L1 distance from the multiplicative identity (1) for spatiotemporal planes.

    Args:
        multi_res_grids: Grids to compute L1 distance over

    Returns:
         L1 distance from the multiplicative identity (1)
    """
    time_planes = [2, 4, 5]  # These are the spatiotemporal planes
    total = 0.0
    num_planes = 0
    for grids in multi_res_grids:
        for grid_id in time_planes:
            total += torch.abs(1 - grids[grid_id]).mean()
            num_planes += 1

    return total / num_planes # type: ignore


def compute_plane_smoothness(t: torch.Tensor) -> float:
    """Computes smoothness across the temporal axis of a plane

    Args:
        t: Plane tensor

    Returns:
        Time smoothness
    """
    _, h, _ = t.shape
    # Convolve with a second derivative filter, in the time dimension which is dimension 2
    first_difference = t[..., 1:, :] - t[..., : h - 1, :]  # [c, h-1, w]
    second_difference = first_difference[..., 1:, :] - first_difference[..., : h - 2, :]  # [c, h-2, w]
    # Take the L2 norm of the result
    return torch.square(second_difference).mean() # type: ignore


def time_smoothness(multi_res_grids: List[torch.Tensor]) -> float:
    """Computes smoothness across each time plane in the grids.

    Args:
        multi_res_grids: Grids to compute time smoothness over

    Returns:
        Time smoothness
    """
    total = 0.0
    num_planes = 0
    for grids in multi_res_grids:
        time_planes = [2, 4, 5]  # These are the spatiotemporal planes
        for grid_id in time_planes:
            total += compute_plane_smoothness(grids[grid_id])
            num_planes += 1

    return total / num_planes