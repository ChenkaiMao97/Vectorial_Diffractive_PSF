import numpy as np
import torch

C_0 = 299792458.13099605
EPSILON_0 = 8.85418782e-12
MU_0 = 1.25663706e-6
dL = 25e-9
n_air = 1.
n_Si = 3.7
n_sub = 1.5

def make_xyz(dl, sx, sy):
    total_length_x = (sx-1) * dl
    total_length_y = (sy-1) * dl
    x = torch.linspace(-total_length_x/2, total_length_x/2, sx)
    y = torch.linspace(-total_length_y/2, total_length_y/2, sy)

    x, y = torch.meshgrid(x, y, indexing='ij')
    x = x.flatten()
    y = y.flatten()
    z = torch.zeros_like(x)
    xyz = torch.stack([x, y, z], dim=1) # (sx*sy, 3)
    return xyz


def StrattonChu(dl, xyz, far_x, far_y, far_z, lambda_val, Ex_top, Ey_top, Ez_top, Hx_top, Hy_top, Hz_top, device='cuda'):
    assert len(Ex_top.shape) == 3 # (bs, sx, sy)
    bs, sx, sy = Ex_top.shape
    
    k0 = 2 * torch.pi / lambda_val
    ds = dl * dl

    xyz = xyz.to(device)

    E_top = torch.stack([Ex_top,Ey_top,Ez_top], dim=3).reshape(bs, -1, 3)
    H_top = torch.stack([Hx_top,Hy_top,Hz_top], dim=3).reshape(bs, -1, 3)

    E = E_top
    H = H_top

    n = torch.tensor([0, 0, -1], device=device, dtype=torch.complex64)
    n = n[None,None,:].repeat(bs, E.shape[1], 1)

    far_xyz = torch.tensor([far_x, far_y, far_z], device=device)

    r_rs = torch.abs(far_xyz - xyz)
    r_rs = torch.sqrt(r_rs[:, 0] ** 2 + r_rs[:, 1] ** 2 + r_rs[:, 2] ** 2)  # [15841]
    ux = (far_x - xyz[:, 0]) / r_rs
    uy = (far_y - xyz[:, 1]) / r_rs
    uz = (far_z - xyz[:, 2]) / r_rs
    t_u = torch.stack([ux, uy, uz], dim=1)
    t_u = t_u.to(torch.complex64)
    t_coe = 1j * k0 * ds * torch.exp(-1j * k0 * r_rs) / (4 * torch.pi * r_rs)
    t_coe = torch.stack([t_coe, t_coe, t_coe], dim=1)
    t_n = n

    t_u = t_u.unsqueeze(0).repeat(bs, 1, 1)
    t_coe = t_coe.unsqueeze(0).repeat(bs, 1, 1)

    far_E = t_coe * (
            -C_0 * MU_0 * torch.cross(t_n, H, dim=-1)
            + torch.cross(torch.cross(t_n, E, dim=-1), t_u, dim=-1)
            + torch.sum(t_n * E, dim=-1).unsqueeze(-1) * t_u
    )
    # far_H = t_coe * (
    #         C_0 * EPSILON_0 * torch.cross(t_n, E, dim=-1)
    #         + torch.cross(torch.cross(t_n, H, dim=-1), t_u, dim=-1)
    #         + torch.sum(t_n * H, dim=-1).unsqueeze(-1) * t_u
    # )

    tg_E_sum = torch.sum(far_E, dim=1)
    return tg_E_sum
    # tg_H_sum = torch.sum(far_H, dim=1)
    # u0 = u0.unsqueeze(0).repeat(bs, 1)

    # return u0, tg_E_sum, tg_H_sum