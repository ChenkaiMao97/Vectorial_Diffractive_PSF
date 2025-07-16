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


def strattonChu3D_GPU(dl, xc, yc, zc, Rx, Ry, Rz, lambda_val, Ex_OBJ, Ey_OBJ, Ez_OBJ, Hx_OBJ, Hy_OBJ, Hz_OBJ,
                      device='cuda', t_theta=0, t_phi=0, n=0, xyz=0, bs=1):
    k0 = 2 * torch.pi / lambda_val
    ds = dl * dl
    r_obs = 10000 * lambda_val

    x_back = int(xc - Rx)
    x_front = int(xc + Rx)
    y_left = int(yc - Ry)
    y_right = int(yc + Ry)
    z_bottom = int(zc - Rz)
    z_top = int(zc + Rz)

    E_null = torch.zeros(bs, 1, 3, device=device, dtype=torch.complex64)
    H_null = torch.zeros(bs, 1, 3, device=device, dtype=torch.complex64)

    E_back = torch.stack([Ex_OBJ[:, x_back, y_left:y_right, z_bottom:z_top],
                          Ey_OBJ[:, x_back, y_left:y_right, z_bottom:z_top],
                          Ez_OBJ[:, x_back, y_left:y_right, z_bottom:z_top]], dim=3).reshape(bs, -1, 3)
    H_back = torch.stack([Hx_OBJ[:, x_back, y_left:y_right, z_bottom:z_top],
                          Hy_OBJ[:, x_back, y_left:y_right, z_bottom:z_top],
                          Hz_OBJ[:, x_back, y_left:y_right, z_bottom:z_top]], dim=3).reshape(bs, -1, 3)

    E_front = torch.stack([Ex_OBJ[:, x_front, y_left:y_right, z_bottom:z_top],
                           Ey_OBJ[:, x_front, y_left:y_right, z_bottom:z_top],
                           Ez_OBJ[:, x_front, y_left:y_right, z_bottom:z_top]], dim=3).reshape(bs, -1, 3)
    H_front = torch.stack([Hx_OBJ[:, x_front, y_left:y_right, z_bottom:z_top],
                           Hy_OBJ[:, x_front, y_left:y_right, z_bottom:z_top],
                           Hz_OBJ[:, x_front, y_left:y_right, z_bottom:z_top]], dim=3).reshape(bs, -1, 3)

    E_left = torch.stack([Ex_OBJ[:, x_back:x_front, y_left, z_bottom:z_top],
                          Ey_OBJ[:, x_back:x_front, y_left, z_bottom:z_top],
                          Ez_OBJ[:, x_back:x_front, y_left, z_bottom:z_top]], dim=3).reshape(bs, -1, 3)
    H_left = torch.stack([Hx_OBJ[:, x_back:x_front, y_left, z_bottom:z_top],
                          Hy_OBJ[:, x_back:x_front, y_left, z_bottom:z_top],
                          Hz_OBJ[:, x_back:x_front, y_left, z_bottom:z_top]], dim=3).reshape(bs, -1, 3)

    E_right = torch.stack([Ex_OBJ[:, x_back:x_front, y_right, z_bottom:z_top],
                           Ey_OBJ[:, x_back:x_front, y_right, z_bottom:z_top],
                           Ez_OBJ[:, x_back:x_front, y_right, z_bottom:z_top]], dim=3).reshape(bs, -1, 3)
    H_right = torch.stack([Hx_OBJ[:, x_back:x_front, y_right, z_bottom:z_top],
                           Hy_OBJ[:, x_back:x_front, y_right, z_bottom:z_top],
                           Hz_OBJ[:, x_back:x_front, y_right, z_bottom:z_top]], dim=3).reshape(bs, -1, 3)

    E_bottom = torch.stack([Ex_OBJ[:, x_back:x_front, y_left:y_right, z_bottom],
                            Ey_OBJ[:, x_back:x_front, y_left:y_right, z_bottom],
                            Ez_OBJ[:, x_back:x_front, y_left:y_right, z_bottom]], dim=3).reshape(bs, -1, 3)
    H_bottom = torch.stack([Hx_OBJ[:, x_back:x_front, y_left:y_right, z_bottom],
                            Hy_OBJ[:, x_back:x_front, y_left:y_right, z_bottom],
                            Hz_OBJ[:, x_back:x_front, y_left:y_right, z_bottom]], dim=3).reshape(bs, -1, 3)

    E_top = torch.stack([Ex_OBJ[:, x_back:x_front, y_left:y_right, z_top],
                         Ey_OBJ[:, x_back:x_front, y_left:y_right, z_top],
                         Ez_OBJ[:, x_back:x_front, y_left:y_right, z_top]], dim=3).reshape(bs, -1, 3)
    H_top = torch.stack([Hx_OBJ[:, x_back:x_front, y_left:y_right, z_top],
                         Hy_OBJ[:, x_back:x_front, y_left:y_right, z_top],
                         Hz_OBJ[:, x_back:x_front, y_left:y_right, z_top]], dim=3).reshape(bs, -1, 3)

    E = torch.cat([E_null, E_back, E_front, E_left, E_right, E_bottom, E_top], dim=1)
    H = torch.cat([H_null, H_back, H_front, H_left, H_right, H_bottom, H_top], dim=1)

    t_theta = t_theta * torch.pi/ 180
    t_phi = t_phi * torch.pi/ 180
    cos_theta = torch.cos(t_theta)
    sin_theta = torch.sin(t_theta)
    cos_phi = torch.cos(t_phi)
    sin_phi = torch.sin(t_phi)
    ux_0 = (cos_phi * sin_theta).to(device)
    uy_0 = (sin_phi * sin_theta).to(device)
    uz_0 = (cos_theta).to(device)
    u0 = torch.stack([ux_0, uy_0, uz_0])

    r_rs = torch.abs(r_obs * u0 - xyz)
    r_rs = torch.sqrt(r_rs[:, 0] ** 2 + r_rs[:, 1] ** 2 + r_rs[:, 2] ** 2)  # [15841]
    ux = (r_obs * ux_0 - xyz[:, 0]) / r_rs
    uy = (r_obs * uy_0 - xyz[:, 1]) / r_rs
    uz = (r_obs * uz_0 - xyz[:, 2]) / r_rs
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
    far_H = t_coe * (
            C_0 * EPSILON_0 * torch.cross(t_n, E, dim=-1)
            + torch.cross(torch.cross(t_n, H, dim=-1), t_u, dim=-1)
            + torch.sum(t_n * H, dim=-1).unsqueeze(-1) * t_u
    )

    tg_E_sum = torch.sum(far_E, dim=1)
    tg_H_sum = torch.sum(far_H, dim=1)
    u0 = u0.unsqueeze(0).repeat(bs, 1)

    return u0, tg_E_sum, tg_H_sum


def strattonChu3D_full_sphere_GPU(dl, xc, yc, zc, Rx, Ry, Rz, lambda_val, Ex_OBJ, Ey_OBJ, Ez_OBJ, Hx_OBJ, Hy_OBJ, Hz_OBJ,
                                  device='cuda', n=0, xyz=0, bs=1):
    k0 = 2 * torch.pi / lambda_val
    ds = dl * dl
    r_obs = 10000 * lambda_val
    
    x_back = int(xc - Rx)
    x_front = int(xc + Rx)
    y_left = int(yc - Ry)
    y_right = int(yc + Ry)
    z_bottom = int(zc - Rz)
    z_top = int(zc + Rz)

    E_null = torch.zeros(bs, 1, 3, device=device, dtype=torch.complex64)
    H_null = torch.zeros(bs, 1, 3, device=device, dtype=torch.complex64)

    E_back = torch.stack([Ex_OBJ[:, x_back, y_left:y_right, z_bottom:z_top],
                          Ey_OBJ[:, x_back, y_left:y_right, z_bottom:z_top],
                          Ez_OBJ[:, x_back, y_left:y_right, z_bottom:z_top]], dim=3).reshape(bs, -1, 3)
    H_back = torch.stack([Hx_OBJ[:, x_back, y_left:y_right, z_bottom:z_top],
                          Hy_OBJ[:, x_back, y_left:y_right, z_bottom:z_top],
                          Hz_OBJ[:, x_back, y_left:y_right, z_bottom:z_top]], dim=3).reshape(bs, -1, 3)

    E_front = torch.stack([Ex_OBJ[:, x_front, y_left:y_right, z_bottom:z_top],
                           Ey_OBJ[:, x_front, y_left:y_right, z_bottom:z_top],
                           Ez_OBJ[:, x_front, y_left:y_right, z_bottom:z_top]], dim=3).reshape(bs, -1, 3)
    H_front = torch.stack([Hx_OBJ[:, x_front, y_left:y_right, z_bottom:z_top],
                           Hy_OBJ[:, x_front, y_left:y_right, z_bottom:z_top],
                           Hz_OBJ[:, x_front, y_left:y_right, z_bottom:z_top]], dim=3).reshape(bs, -1, 3)

    E_left = torch.stack([Ex_OBJ[:, x_back:x_front, y_left, z_bottom:z_top],
                          Ey_OBJ[:, x_back:x_front, y_left, z_bottom:z_top],
                          Ez_OBJ[:, x_back:x_front, y_left, z_bottom:z_top]], dim=3).reshape(bs, -1, 3)
    H_left = torch.stack([Hx_OBJ[:, x_back:x_front, y_left, z_bottom:z_top],
                          Hy_OBJ[:, x_back:x_front, y_left, z_bottom:z_top],
                          Hz_OBJ[:, x_back:x_front, y_left, z_bottom:z_top]], dim=3).reshape(bs, -1, 3)

    E_right = torch.stack([Ex_OBJ[:, x_back:x_front, y_right, z_bottom:z_top],
                           Ey_OBJ[:, x_back:x_front, y_right, z_bottom:z_top],
                           Ez_OBJ[:, x_back:x_front, y_right, z_bottom:z_top]], dim=3).reshape(bs, -1, 3)
    H_right = torch.stack([Hx_OBJ[:, x_back:x_front, y_right, z_bottom:z_top],
                           Hy_OBJ[:, x_back:x_front, y_right, z_bottom:z_top],
                           Hz_OBJ[:, x_back:x_front, y_right, z_bottom:z_top]], dim=3).reshape(bs, -1, 3)

    E_bottom = torch.stack([Ex_OBJ[:, x_back:x_front, y_left:y_right, z_bottom],
                            Ey_OBJ[:, x_back:x_front, y_left:y_right, z_bottom],
                            Ez_OBJ[:, x_back:x_front, y_left:y_right, z_bottom]], dim=3).reshape(bs, -1, 3)
    H_bottom = torch.stack([Hx_OBJ[:, x_back:x_front, y_left:y_right, z_bottom],
                            Hy_OBJ[:, x_back:x_front, y_left:y_right, z_bottom],
                            Hz_OBJ[:, x_back:x_front, y_left:y_right, z_bottom]], dim=3).reshape(bs, -1, 3)

    E_top = torch.stack([Ex_OBJ[:, x_back:x_front, y_left:y_right, z_top],
                         Ey_OBJ[:, x_back:x_front, y_left:y_right, z_top],
                         Ez_OBJ[:, x_back:x_front, y_left:y_right, z_top]], dim=3).reshape(bs, -1, 3)
    H_top = torch.stack([Hx_OBJ[:, x_back:x_front, y_left:y_right, z_top],
                         Hy_OBJ[:, x_back:x_front, y_left:y_right, z_top],
                         Hz_OBJ[:, x_back:x_front, y_left:y_right, z_top]], dim=3).reshape(bs, -1, 3)

    E = torch.cat([E_null, E_back, E_front, E_left, E_right, E_bottom, E_top], dim=1) # (bs, 15841, 3)
    H = torch.cat([H_null, H_back, H_front, H_left, H_right, H_bottom, H_top], dim=1) # (bs, 15841, 3)

    theta_step = 1
    theta_min, theta_max = theta_step, 180 - theta_step
    theta_range = torch.linspace(theta_min, theta_max, (theta_max - theta_min) // theta_step + 1)
    phi_step = 4
    phi_min, phi_max = 0, 360 - phi_step
    phi_range = torch.linspace(phi_min, phi_max, (phi_max - phi_min) // phi_step + 1)
    N_theta, N_phi = len(theta_range), len(phi_range)

    thetas, phis = torch.meshgrid(theta_range, phi_range, indexing='ij')
    thetas = (thetas * torch.pi/ 180).to(device)
    phis = (phis * torch.pi/ 180).to(device)
    cos_theta = torch.cos(thetas)
    sin_theta = torch.sin(thetas)
    cos_phi = torch.cos(phis)
    sin_phi = torch.sin(phis)
    ux_0 = cos_phi * sin_theta
    uy_0 = sin_phi * sin_theta
    uz_0 = cos_theta
    u0 = torch.stack([ux_0, uy_0, uz_0]) # (3, N_theta, N_phi)

    # xyz: (15841, 3)
    r_rs = torch.abs(r_obs * u0[None] - xyz[:,:,None,None]) # (15841, 3, N_theta, N_phi)
    r_rs = torch.sqrt(r_rs[:, 0] ** 2 + r_rs[:, 1] ** 2 + r_rs[:, 2] ** 2)  # [15841, N_theta, N_phi]
    ux = (r_obs * ux_0[None] - xyz[:, 0, None, None]) / r_rs
    uy = (r_obs * uy_0[None] - xyz[:, 1, None, None]) / r_rs
    uz = (r_obs * uz_0[None] - xyz[:, 2, None, None]) / r_rs
    t_u = torch.stack([ux, uy, uz], dim=1).to(torch.complex64) # (15841, 3, N_theta, N_phi)
    t_coe = 1j * k0 * ds * torch.exp(-1j * k0 * r_rs) / (4 * torch.pi * r_rs) # (15841, N_theta, N_phi)
    t_coe = torch.stack([t_coe, t_coe, t_coe], dim=1) # (15841, 3, N_theta, N_phi)
    t_n = n[...,None] # (bs, 15841, 3, 1)

    t_u = t_u.unsqueeze(0).repeat(bs, 1, 1, 1, 1) # (bs, 15841, 3, N_theta, N_phi)
    t_coe = t_coe.unsqueeze(0).repeat(bs, 1, 1, 1, 1) # (bs, 15841, 3, N_theta, N_phi)


    #######################  processing theta iteratively to avoid OOM ####################### 
    tg_E_sum = torch.zeros(bs, 3, N_theta, N_phi, device=device, dtype=torch.complex64)
    tg_H_sum = torch.zeros(bs, 3, N_theta, N_phi, device=device, dtype=torch.complex64)
    for theta_idx in range(N_theta):
        far_E = t_coe[:,:,:,theta_idx,:] * (
                -C_0 * MU_0 * torch.cross(t_n, H[...,None], dim=2)
                + torch.cross(torch.cross(t_n, E[...,None], dim=2), t_u[:,:,:,theta_idx,:], dim=2)
                + torch.sum(t_n * E[...,None], dim=2, keepdim=True) * t_u[:,:,:,theta_idx,:]
        )
        far_H = t_coe[:,:,:,theta_idx,:] * (
                C_0 * EPSILON_0 * torch.cross(t_n, E[...,None], dim=2)
                + torch.cross(torch.cross(t_n, H[...,None], dim=2), t_u[:,:,:,theta_idx,:], dim=2)
                + torch.sum(t_n * H[...,None], dim=2, keepdim=True) * t_u[:,:,:,theta_idx,:]
        )
        far_E_sum = torch.sum(far_E, dim=1)
        far_H_sum = torch.sum(far_H, dim=1)
        tg_E_sum[:,:,theta_idx,:] = far_E_sum
        tg_H_sum[:,:,theta_idx,:] = far_H_sum
    #######################  processing all directions at once ####################### 
    # far_E = t_coe * (
    #         -C_0 * MU_0 * torch.cross(t_n, H[...,None,None], dim=2)
    #         + torch.cross(torch.cross(t_n, E[...,None,None], dim=2), t_u, dim=2)
    #         + torch.sum(t_n * E[...,None,None], dim=2, keepdim=True) * t_u
    # )
    # far_H = t_coe * (
    #         C_0 * EPSILON_0 * torch.cross(t_n, E[...,None,None], dim=2)
    #         + torch.cross(torch.cross(t_n, H[...,None,None], dim=2), t_u, dim=2)
    #         + torch.sum(t_n * H[...,None,None], dim=2, keepdim=True) * t_u
    # )
    # tg_E_sum = torch.sum(far_E, dim=1) # (bs, 3, N_theta, N_phi)
    # tg_H_sum = torch.sum(far_H, dim=1) # (bs, 3, N_theta, N_phi)
    ##################################################################################
    u0 = u0.unsqueeze(0).repeat(bs, 1, 1, 1) # (bs, 3, N_theta, N_phi)

    return thetas, phis, u0, tg_E_sum, tg_H_sum