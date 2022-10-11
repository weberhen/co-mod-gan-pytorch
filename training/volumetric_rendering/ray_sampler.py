# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
The ray sampler is a module that takes in camera matrices and resolution and batches of rays.
Expects cam2world matrices that use the OpenCV camera coordinate system conventions.
"""

import torch

class RaySampler(torch.nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.ray_origins_h, self.ray_directions, self.depths, self.image_coords, self.rendering_options = None, None, None, None, None
        self.device = device

    def forward(self, cam2world_matrix, resolution_x, resolution_y, random_ray_origin=True, N=1):
        """
        Create batches of rays and return origins and directions.

        resolution: int

        ray_origins: (N, M, 3)
        ray_dirs: (N, M, 3)
        """
        # Compute coordinates of pixel borders
        cols = (torch.arange(0, resolution_x, dtype=torch.float32, device=self.device) + .5) / resolution_x
        rows = (torch.arange(0, resolution_y, dtype=torch.float32, device=self.device) + .5) / resolution_y

        v, u = torch.meshgrid(cols, rows)

        u = u * 2

        # lat-long -> world
        thetaLatLong = torch.pi * (u - 1)
        phiLatLong = torch.pi * v

        x = (torch.sin(phiLatLong) * torch.sin(thetaLatLong)).flatten().to(cam2world_matrix.device)
        y = (torch.cos(phiLatLong)).flatten().to(cam2world_matrix.device)
        z = (-torch.sin(phiLatLong) * torch.cos(thetaLatLong)).flatten().to(cam2world_matrix.device)

        M = x.shape[0]
        # ray_origins 
        ray_origins = torch.zeros(N, M, 3).to(cam2world_matrix.device)
        
        if random_ray_origin:
            # create some random displacement value between -.25 and .25 for x
            x_displacement = (torch.rand(1).to(cam2world_matrix.device) * .5 - .25).repeat(x.shape[0])
            # create some random displacement value between -.25 and .25 for z
            z_displacement = (torch.rand(1).to(cam2world_matrix.device) * .5 - .25).repeat(x.shape[0])
        else:
            x_displacement = cam2world_matrix[0,0,3]
            z_displacement = cam2world_matrix[0,2,3]
        
        ray_origins[:,:,0] = x_displacement
        ray_origins[:,:,2] = z_displacement

        # flatten to (M, 3) then expand to (N, M, 3) 
        ray_dirs = torch.stack([x, y, z], dim=1).expand(N, -1, -1)

        # normalize ray_dirs
        ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=2)

        return ray_origins, ray_dirs