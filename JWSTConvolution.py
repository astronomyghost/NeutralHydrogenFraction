from astropy.io import fits as pyfits
import numpy as np
from scipy.interpolate import interp1d

def convolve_spectrum_prism(lam_obs, flam_obs):
    """
    lam_obs : observed wavelength in microns.
    flam_obs : observed flux array in FLAM units
    """
    f = pyfits.open('lineplot_sn.fits')
    lam_nirspec = f[1].data['WAVELENGTH']
    f.close()
    
    delta_wl_low = np.diff(lam_nirspec)
    wl_center = 0.5 * (lam_nirspec[:-1] + lam_nirspec[1:])
    R_low = wl_center / delta_wl_low  # R = λ / Δλ
    R_interp = interp1d(wl_center, R_low, bounds_error=False, fill_value="extrapolate")
    R_high = R_interp(lam_obs)

    delta_wl_high = np.diff(lam_obs)
    wl_center_high = 0.5 * (lam_obs[:-1] + lam_obs[1:])
    sigma_lambda = np.sqrt((lam_obs / R_high) ** 2 - delta_wl_high.mean() ** 2)
    sigma_lambda[sigma_lambda < 0] = 0  # Avoid negative values
    sigma_pix = sigma_lambda / np.gradient(lam_obs)
    
    flux_convolved = np.zeros_like(flam_obs)
    for i in range(len(lam_obs)):
        wl_i = lam_obs[i]
        sigma_i = sigma_lambda[i]
        
        if sigma_i <= 0:  # Avoid negative or zero sigma
            flux_convolved[i] = flam_obs[i]
            continue

        # Define Gaussian kernel
        kernel = np.exp(-0.5 * ((lam_obs - wl_i) / sigma_i) ** 2)
        kernel /= np.sum(kernel)  # Normalize
        
        # Apply convolution at each point
        flux_convolved[i] = np.sum(flam_obs * kernel)
    convolved_spec = interp1d(lam_obs, flux_convolved, bounds_error=False, fill_value="extrapolate")(lam_nirspec)
    return lam_nirspec, convolved_spec