import os, sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from main import get_farfield

from analytical import make_airy_disk, make_sinc_function

plt.rcParams.update({
    'font.size': 20,               # Default font size
    # 'font.family': 'Aptos',        # Default font family
    'font.sans-serif': ['Arial'],  # Specific font for sans-serif
    'axes.titlesize': 20,             # Font size for axes titles
    'axes.labelsize': 20,             # Font size for x and y labels
    'xtick.labelsize': 20,            # Font size for x-axis tick labels
    'ytick.labelsize': 20,            # Font size for y-axis tick labels
    'legend.fontsize': 16,            # Font size for legend
    'figure.titlesize': 20
})


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    n_oil = config['n_oil']
    aperture_side_length = config['aperture_side_length']
    aperture_size = (aperture_side_length, aperture_side_length)
    dl = config['dl']
    wavelength = config['wavelength']
    farfield_window_size = config['farfield_window_size']
    farfield_dl = config['farfield_dl']
    aperture_shape = config['aperture_shape']
    design_type = config['design_type']

    SC_FWHM_list = []
    analytical_FWHM_list = []
    NA_list = [i*0.1 for i in range(10, 11)]
    for NA in NA_list:
        sinth = NA/n_oil
        tanth = sinth / np.sqrt(1 - sinth**2)
        focal_length = aperture_side_length / (2 * tanth)
        print(f"NA: {NA}, focal_length: {focal_length*1e6:.2f} um")

        SC_FWHM, analytical_FWHM = get_farfield(farfield_window_size, farfield_dl, aperture_size, dl, wavelength, focal_length, n_oil, design_type=design_type, aperture_shape=aperture_shape, device=device, save_path="./test")

        SC_FWHM_list.append(SC_FWHM)
        analytical_FWHM_list.append(analytical_FWHM)

    # np.save(os.path.join('test', 'NA_list.npy'), NA_list)
    # np.save(os.path.join('test', 'SC_FWHM_list.npy'), SC_FWHM_list)
    # np.save(os.path.join('test', 'analytical_FWHM_list.npy'), analytical_FWHM_list)

    # plt.plot(NA_list, SC_FWHM_list, label='Stratton-Chu FWHM')
    # plt.plot(NA_list, analytical_FWHM_list, label='Analytical FWHM')
    # plt.yscale('log')
    # plt.xlabel('NA')
    # plt.ylabel('FWHM (um)')
    # plt.legend()
    # plt.savefig('compare_with_analytical.png', dpi=300)