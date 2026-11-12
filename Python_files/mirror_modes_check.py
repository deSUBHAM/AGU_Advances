import happi
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
# import scipy.constants as scon
# from scipy.optimize import curve_fit
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
# from scipy.interpolate import griddata
# from scipy.signal import convolve2d
# from scipy.ndimage import gaussian_filter, laplace
from itertools import cycle
import time
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as patches
import matplotlib.ticker as tkr
# from scipy.integrate import simps
# from scipy.integrate import quad
import matplotlib.cm as cm
import matplotlib.colors
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import curve_fit
from scipy.optimize import least_squares

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None

###############################################################


def orient_to_NyNx(arr2d, x_axis, y_axis, name="arr"):
    """
    Return arr in shape (Ny, Nx).
    Accepts either (Ny, Nx) or (Nx, Ny).
    """
    arr2d = np.asarray(arr2d)
    Nx, Ny = len(x_axis), len(y_axis)

    if arr2d.shape == (Ny, Nx):
        return arr2d
    if arr2d.shape == (Nx, Ny):
        return arr2d.T

    raise ValueError(
        f"{name} has shape {arr2d.shape}, expected (Ny,Nx)=({Ny},{Nx}) or (Nx,Ny)=({Nx},{Ny})."
    )


def make_B0_background(Bx, By, Bz, sigma_bg=10.0):
    """
    Compute a local background field B0 via Gaussian smoothing.
    sigma_bg is in grid cells.
    """
    if gaussian_filter is None or sigma_bg is None or sigma_bg <= 0:
        # fallback: no smoothing
        return Bx.copy(), By.copy(), Bz.copy()

    B0x = gaussian_filter(Bx, sigma=sigma_bg)
    B0y = gaussian_filter(By, sigma=sigma_bg)
    B0z = gaussian_filter(Bz, sigma=sigma_bg)
    return B0x, B0y, B0z

###############################################################
def field_aligned_pressures_from_diagonals(Pxx, Pyy, Pzz, B0x, B0y, B0z,
                                          eps=1e-30, min_B0_frac=0.05):
    """
    Approximate P_parallel and P_perp using only diagonal components of P tensor.
    Works best if off-diagonal terms are small or if you only want a robust proxy.

    Returns:
      Ppar, Pperp, B0mag, valid_mask
    """
    Pxx = np.asarray(Pxx); Pyy = np.asarray(Pyy); Pzz = np.asarray(Pzz)
    B0x = np.asarray(B0x); B0y = np.asarray(B0y); B0z = np.asarray(B0z)

    B0mag = np.sqrt(B0x**2 + B0y**2 + B0z**2)
    maxB0 = np.nanmax(B0mag) if np.isfinite(B0mag).any() else 0.0
    valid = B0mag > (min_B0_frac * maxB0)

    # unit b0
    bhx = np.zeros_like(B0x); bhy = np.zeros_like(B0y); bhz = np.zeros_like(B0z)
    bhx[valid] = B0x[valid] / (B0mag[valid] + eps)
    bhy[valid] = B0y[valid] / (B0mag[valid] + eps)
    bhz[valid] = B0z[valid] / (B0mag[valid] + eps)

    # P_parallel ≈ bx^2 Pxx + by^2 Pyy + bz^2 Pzz   (diagonal-only approximation)
    Ppar = (bhx**2)*Pxx + (bhy**2)*Pyy + (bhz**2)*Pzz

    Ptr = Pxx + Pyy + Pzz
    Pperp = 0.5*(Ptr - Ppar)

    return Ppar, Pperp, B0mag, valid


def betas_from_pressures(Ppar, Pperp, B0mag, eps=1e-30):
    Pmag0 = 0.5*(B0mag**2)  # your normalization uses B^2/2
    beta_par  = Ppar  / (Pmag0 + eps)
    beta_perp = Pperp / (Pmag0 + eps)
    return beta_par, beta_perp


def mirror_instability_parameter(beta_par, beta_perp):
    # CGL-style mirror threshold written as: beta_par*(1+beta_perp) - beta_perp^2 < 0
    return beta_par*(1.0 + beta_perp) - beta_perp**2
###############################################################
def magnetic_compressibility(Bx, By, Bz, B0x, B0y, B0z,
                             eps=1e-30, min_B0_frac=0.05):
    """
    Compute:
      C_B    = dB_parallel^2 / |dB|^2   (relative to local background direction b0)
      C_|B|  = (d|B|)^2 / |dB|^2
    Returns: C_B, C_absB, valid_mask
    """
    Bx = np.asarray(Bx); By = np.asarray(By); Bz = np.asarray(Bz)
    B0x = np.asarray(B0x); B0y = np.asarray(B0y); B0z = np.asarray(B0z)

    Bmag  = np.sqrt(Bx**2 + By**2 + Bz**2)
    B0mag = np.sqrt(B0x**2 + B0y**2 + B0z**2)

    maxB0 = np.nanmax(B0mag) if np.isfinite(B0mag).any() else 0.0
    valid = B0mag > (min_B0_frac * maxB0)

    # unit b0
    bhx = np.zeros_like(B0x); bhy = np.zeros_like(B0y); bhz = np.zeros_like(B0z)
    bhx[valid] = B0x[valid] / (B0mag[valid] + eps)
    bhy[valid] = B0y[valid] / (B0mag[valid] + eps)
    bhz[valid] = B0z[valid] / (B0mag[valid] + eps)

    dBx = Bx - B0x
    dBy = By - B0y
    dBz = Bz - B0z

    dB2 = dBx**2 + dBy**2 + dBz**2
    dBpar = dBx*bhx + dBy*bhy + dBz*bhz

    # C_B
    Cb = np.full_like(dB2, np.nan, dtype=float)
    Cb[valid] = (dBpar[valid]**2) / (dB2[valid] + eps)

    # C_|B|
    dBmag = Bmag - B0mag
    CabsB = np.full_like(dB2, np.nan, dtype=float)
    CabsB[valid] = (dBmag[valid]**2) / (dB2[valid] + eps)

    return Cb, CabsB, valid, Bmag, B0mag

###############################################################
def local_corr_map(a, b, sigma_corr=10.0, eps=1e-30):
    """
    Local correlation coefficient corr(a,b) computed via Gaussian window.
    Returns corr map in [-1,1] (NaN where variance is too small).
    """
    a = np.asarray(a); b = np.asarray(b)
    if gaussian_filter is None or sigma_corr is None or sigma_corr <= 0:
        raise RuntimeError("scipy.ndimage.gaussian_filter is required for local_corr_map")

    a0 = gaussian_filter(a, sigma=sigma_corr)
    b0 = gaussian_filter(b, sigma=sigma_corr)
    da = a - a0
    db = b - b0

    cov = gaussian_filter(da*db, sigma=sigma_corr)
    va  = gaussian_filter(da*da, sigma=sigma_corr)
    vb  = gaussian_filter(db*db, sigma=sigma_corr)

    corr = cov / np.sqrt(va*vb + eps)
    return corr

