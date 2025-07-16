import numpy as np
import scipy.special as sp

def sinc(x):
    if abs(x) < 1e-6:
        return 1
    return np.sin(x)/(x)

def make_airy_disk(farfield_window_size, farfield_dl, aperture_size, wavelength, focal_length, oil_index):
    window_pixels = round(farfield_window_size / farfield_dl)
    wl = wavelength / oil_index
    k = 2 * np.pi / wl
    diameter = aperture_size[0]
    
    psf = np.zeros((window_pixels, window_pixels))
    for i in range(window_pixels):
        for j in range(window_pixels):
            x = (i - (window_pixels-1)/2)*farfield_dl
            y = (j - (window_pixels-1)/2)*farfield_dl
            r = (x**2 + y**2) **.5
            kr = k * diameter * r / (2 * focal_length)

            J1 = sp.j1(kr)

            psf[i,j] = (2 * J1 / kr)**2
    
    return psf/np.sum(psf)


def make_sinc_function(farfield_window_size, farfield_dl, aperture_size, wavelength, focal_length, oil_index):
    window_pixels = round(farfield_window_size / farfield_dl)
    wl = wavelength / oil_index
    
    psf = np.zeros((window_pixels, window_pixels))
    for i in range(window_pixels):
        for j in range(window_pixels):
            x = (i - (window_pixels-1)/2)*farfield_dl
            y = (j - (window_pixels-1)/2)*farfield_dl

            psf[i,j] = sinc(np.pi*aperture_size[0]*x/wl/focal_length)**2 * sinc(np.pi*aperture_size[1]*y/wl/focal_length)**2
    
    return psf/np.sum(psf)
