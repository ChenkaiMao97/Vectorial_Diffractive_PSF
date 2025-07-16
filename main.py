import os, sys
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml

from analytical import make_airy_disk, make_sinc_function

from stratton_chu_GPU_single import StrattonChu, make_xyz


C_0 = 299792458.13099605
EPSILON_0 = 8.85418782e-12
MU_0 = 1.25663706e-6
dL = 25e-9
n_air = 1.
n_Si = 3.7
n_sub = 1.5

def FWHM_line(line, dl, level=0.5):
    # print("resolution: ", dl)
    vm = np.max(line)
    HM = level*vm

    for i in range(1, len(line)):
        if line[i] >=HM and line[i-1] < HM:
            left = (i-1) + (HM-line[i-1])/(line[i]-line[i-1])
        if line[i] <=HM and line[i-1] > HM:
            right = i - (HM-line[i])/(line[i-1]-line[i])
    # print("left: ", left, "right: ", right)
    return float((right - left) * dl)

def phase_lens(x, y, wavelength_in_medium, focal_length):
    r_square = x**2 + y**2
    distance = torch.sqrt(focal_length**2 + r_square)
    phase = 2 * np.pi / wavelength_in_medium * (distance - focal_length)
    return phase

def make_top_field(aperture_size, dl, wavelength, focal_length, oil_index, design_type='geometric', aperture_shape='square'):
    sx, sy = aperture_size
    pixel_x, pixel_y = round(sx / dl) + 1, round(sy / dl) + 1

    x = torch.linspace(-aperture_size[0]/2, aperture_size[0]/2, pixel_x)
    y = torch.linspace(-aperture_size[1]/2, aperture_size[1]/2, pixel_y)

    x, y = torch.meshgrid(x, y, indexing='ij')
    phase = phase_lens(x, y, wavelength/oil_index, focal_length)

    impedance = np.sqrt(MU_0 / EPSILON_0) * 1/np.sqrt(oil_index)
    if design_type == 'geometric':
        Ex = torch.exp(1j * phase)
        Ey = torch.exp(1j * (phase - np.pi/2))
        Ez = torch.zeros_like(Ex)

        Hx = - Ey / impedance
        Hy = Ex / impedance
        Hz = torch.zeros_like(Ex)

    elif design_type == 'linear_polarized':
        Ex = torch.exp(1j * phase)
        Ey = torch.zeros_like(Ex)
        Ez = torch.zeros_like(Ex)

        Hx = torch.zeros_like(Ex)
        Hy = Ex / impedance
        Hz = torch.zeros_like(Ex)
    else:
        raise ValueError(f"Design type {design_type} is not supported")
    
    if aperture_shape == 'circle':
        mask = (x**2 + y**2) <= aperture_size[0]**2/4
        Ex = Ex * mask
        Ey = Ey * mask
        Ez = Ez * mask
        Hx = Hx * mask
        Hy = Hy * mask
        Hz = Hz * mask
    elif aperture_shape == 'square':
        pass
    else:
        raise ValueError(f"Aperture shape {aperture_shape} is not supported")

    return Ex, Ey, Ez, Hx, Hy, Hz

def get_farfield(farfield_window_size, farfield_dl, aperture_size, dl, wavelength, focal_length, oil_index, design_type='geometric', aperture_shape='square', device=torch.device('cpu'), save_path=None):
    Ex, Ey, Ez, Hx, Hy, Hz = make_top_field(aperture_size, dl, wavelength, focal_length, oil_index, design_type, aperture_shape)
    Ex, Ey, Ez, Hx, Hy, Hz = Ex[None].to(device), Ey[None].to(device), Ez[None].to(device), Hx[None].to(device), Hy[None].to(device), Hz[None].to(device)
    _, sx, sy = Ex.shape

    farfield_pixels = round(farfield_window_size / farfield_dl)
    far_z = focal_length
    
    xyz = make_xyz(dl, sx, sy).to(device)
    E_farfield = torch.zeros((farfield_pixels, farfield_pixels, 3), device=device, dtype=torch.complex64)
    for i in tqdm(range(farfield_pixels)):
        for j in range(farfield_pixels):
            far_x = (-farfield_pixels/2 + i) * farfield_dl
            far_y = (-farfield_pixels/2 + j) * farfield_dl

            e_farfield = StrattonChu(dl, xyz, far_x, far_y, far_z, wavelength/oil_index, Ex, Ey, Ez, Hx, Hy, Hz, device=device)
            E_farfield[i, j, :] = e_farfield[0]
    
    E_farfield = E_farfield.cpu().numpy()
    intensity = np.abs(E_farfield[:,:,0])**2 + np.abs(E_farfield[:,:,1])**2 + np.abs(E_farfield[:,:,2])**2
    intensity = intensity / np.sum(intensity)
    
    if aperture_shape == 'circle':
        analytical = 'Airy disk'
        analytical_intensity = make_airy_disk(farfield_window_size, farfield_dl, aperture_size, wavelength, focal_length, oil_index)
    elif aperture_shape == 'square':
        analytical = 'Sinc function'
        analytical_intensity = make_sinc_function(farfield_window_size, farfield_dl, aperture_size, wavelength, focal_length, oil_index)
    else:
        raise ValueError(f"Aperture shape {aperture_shape} is not supported")

    far_x = np.linspace(-farfield_window_size/2, farfield_window_size/2, farfield_pixels)
    far_y = np.linspace(-farfield_window_size/2, farfield_window_size/2, farfield_pixels)
    far_x, far_y =  np.meshgrid(far_x*1e6, far_y*1e6, indexing='ij')
    
    if save_path is not None:
        np.save(os.path.join(save_path, f'intensity_far_z_{far_z*1e6:.2f}um.npy'), intensity)
        plt.figure(figsize=(26, 4))
        plt.subplot(1, 5, 1)
        plt.pcolormesh(far_x, far_y, np.abs(E_farfield[:,:,0]), cmap='turbo')
        plt.gca().set_aspect('equal')
        plt.xlabel('x (μm)')
        plt.ylabel('y (μm)')
        plt.colorbar()
        plt.title('farfield Ex')
        plt.subplot(1, 5, 2)
        plt.pcolormesh(far_x, far_y, np.abs(E_farfield[:,:,1]), cmap='turbo')
        plt.gca().set_aspect('equal')
        plt.xlabel('x (μm)')
        plt.ylabel('y (μm)')
        plt.colorbar()
        plt.title('farfield Ey')
        plt.subplot(1, 5, 3)
        plt.pcolormesh(far_x, far_y, np.abs(E_farfield[:,:,2]), cmap='turbo')
        plt.gca().set_aspect('equal')
        plt.xlabel('x (μm)')
        plt.ylabel('y (μm)')
        plt.colorbar()
        plt.title('farfield Ez')
        plt.subplot(1, 5, 4)
        plt.pcolormesh(far_x, far_y, intensity, cmap='turbo')
        plt.gca().set_aspect('equal')
        plt.xlabel('x (μm)')
        plt.ylabel('y (μm)')
        # plt.colorbar()
        # plt.title('farfield intensity (normalized)')
        plt.subplot(1, 5, 5)
        plt.pcolormesh(far_x, far_y, analytical_intensity, cmap='turbo')
        plt.gca().set_aspect('equal')
        plt.xlabel('x (μm)')
        plt.ylabel('y (μm)')
        # plt.colorbar()
        # plt.title(f'intensity from Fraunhofer diffraction\n({analytical})')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'intensity_far_z_{far_z*1e6:.2f}um.png'), dpi=300)
        plt.close()
    
    SC_FWHM = FWHM_line(intensity[:, farfield_pixels//2], farfield_dl)
    analytical_FWHM = FWHM_line(analytical_intensity[:, farfield_pixels//2], farfield_dl)
    print(f"FWHM for stratton-chu: {SC_FWHM*1e6:.4f} um")
    print(f"FWHM for analytical: {analytical_FWHM*1e6:.4f} um")
    return SC_FWHM, analytical_FWHM

if __name__ == "__main__":
    config_file = sys.argv[1]
    save_path = os.path.dirname(config_file)
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    n_oil = config['n_oil']
    NA = config['NA']
    aperture_side_length = config['aperture_side_length']
    aperture_size = (aperture_side_length, aperture_side_length)
    dl = config['dl']
    wavelength = config['wavelength']
    farfield_window_size = config['farfield_window_size']
    farfield_dl = config['farfield_dl']
    aperture_shape = config['aperture_shape']
    design_type = config['design_type']

    sinth = NA/n_oil
    tanth = sinth / np.sqrt(1 - sinth**2)
    focal_length = aperture_side_length / (2 * tanth)
    print(f"focal_length: {focal_length*1e6:.2f} um")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    get_farfield(farfield_window_size, farfield_dl, aperture_size, dl, wavelength, focal_length, n_oil, design_type=design_type, aperture_shape=aperture_shape, device=device, save_path=save_path)
