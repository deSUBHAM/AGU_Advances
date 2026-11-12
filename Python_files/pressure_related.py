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


# ---------- name -> particle-binning-index mapping ----------
# Update this mapping if you create/remove diagnostics.
PARTICLE_MAP = {
    "e_n": 0,  "e_px": 1,  "e_py": 2,  "e_pz": 3,  "e_rho": 4,
    "e_jx": 5, "e_jy": 6,   "e_jz": 7,  "e_ekin": 8,
    "e_pxx": 9,"e_pyy": 10, "e_pzz": 11, "e_pxy": 12, "e_pyz": 13, "e_pxz": 14,
    "i_n": 15, "i_px": 16,  "i_py": 17, "i_pz": 18, "i_rho": 19,
    "i_jx": 20,"i_jy": 21,  "i_jz": 22, "i_ekin": 23,
    "i_pxx": 24,"i_pyy": 25, "i_pzz": 26, "i_pxy": 27, "i_pyz": 28, "i_pxz": 29
}

# ---------- helper to try to extract diagnostic axes ----------
def _try_get_axes(diag):
    """Return (x_axis, y_axis) if possible, else (None, None)."""
    # Many happi diagnostics offer .getAxes() or .axes attributes; attempt them.
    try:
        axes = diag.getAxes()            # some diagnostics
        # getAxes() may return dict like {'x': array, 'y': array}
        if isinstance(axes, dict):
            return axes.get('x', None), axes.get('y', None)
    except Exception:
        pass
    for attr in ("axes", "axes_dict", "axis"):
        if hasattr(diag, attr):
            try:
                a = getattr(diag, attr)
                if isinstance(a, dict):
                    return a.get('x', None), a.get('y', None)
            except Exception:
                pass
    # Not found
    return None, None

# ---------- low-level fetcher ----------
def _fetch_particle_diag(S, idx, frame, time_steps, average=None):
    """
    Fetches diag idx from S.ParticleBinning at time_steps[frame].
    average: None (2D) or {"y":"all"} (1D avg over y)
    Returns (data_array, x_axis_arr, y_axis_arr, diag_object)
    """
    By1      = S.Field(0,'By')
    time_step  = By1.getTimesteps()
    ts = time_step[frame]
    if average is None:
        diag = S.ParticleBinning(idx, timesteps=ts)
    else:
        diag = S.ParticleBinning(idx, timesteps=ts, average=average)
    raw = np.array(diag.getData())   # usually shape (1, nx, ny) or (1, nx) for averaged
    data = np.squeeze(raw)           # squeeze singleton dims
    # attempt to obtain axes from diag; fallback later
    x_axis, y_axis = _try_get_axes(diag)
    # fallback axes if not available
    if x_axis is None:
        if data.ndim == 1:
            x_axis = np.arange(data.size)
        elif data.ndim == 2:
            x_axis = np.arange(data.shape[0])
        else:
            x_axis = None
    if y_axis is None:
        if data.ndim == 2:
            y_axis = np.arange(data.shape[1])
        else:
            y_axis = None
    return data, x_axis, y_axis, diag

# ---------- function 1: full 2D data at a frame ----------
def particle_2d_fields(S, frame, time_steps, var_list=None):
    """
    Load full 2D ParticleBinning diagnostics at a chosen frame.
    Inputs:
      S           : happi S object
      frame       : integer index into time_steps
      time_steps  : array-like of diagnostic time steps / iteration numbers
      var_list    : list of variable keys (strings) from PARTICLE_MAP; if None, load a default subset.
    Returns:
      results: dict with keys = var names -> {"data": 2D numpy array, "x": xaxis, "y": yaxis}
    Example var names: 'e_n','e_jx','i_ekin','i_px', ...
    """
    if var_list is None:
        # default subset you may change
        var_list = ["e_n","e_jx","e_ekin","i_n","i_jx","i_ekin"]
    results = {}
    for name in var_list:
        if name not in PARTICLE_MAP:
            raise KeyError(f"Unknown variable '{name}'. Update PARTICLE_MAP.")
        idx = PARTICLE_MAP[name]
        data, xa, ya, diag = _fetch_particle_diag(S, idx, frame, time_steps, average=None)
        results[name] = {"data": data, "x": xa, "y": ya, "diag": diag}
    return results

# ---------- function 2: y-averaged 1D data at a frame ----------
def particle_1d_yavg(S, frame, time_steps, var_list=None):
    """
    Load ParticleBinning diagnostics averaged over y (average={"y":"all"}).
    Inputs same as particle_2d_fields.
    Returns:
    results: dict name -> {"data": 1D numpy array, "x": xaxis, "diag": diag}
    """
    if var_list is None:
        var_list = list(PARTICLE_MAP.keys())   # load all by default (might be many)
    results = {}
    for name in var_list:
        if name not in PARTICLE_MAP:
            raise KeyError(f"Unknown variable '{name}'. Update PARTICLE_MAP.")
        idx = PARTICLE_MAP[name]
        data, xa, ya, diag = _fetch_particle_diag(S, idx, frame, time_steps, average={"y":"all"})
        # data should be 1D (x)
        if data.ndim != 1:
            # try to squeeze further if accidental dims remain
            data = np.squeeze(data)
            if data.ndim != 1:
                raise ValueError(f"1D average returned array with shape {data.shape} for {name}")
                results[name] = {"data": data, "x": xa, "diag": diag}
            return results


def particle_1d_yavg_safe(S, frame, time_steps, var_list=None):
    """
    Robust 1D y-averaged ParticleBinning loader for Smilei.
    - Skips diagnostics that do not exist.
    - Warns the user for missing diagnostics or missing data.
    - Always returns a dict with keys = requested variables; values = dict with keys "data", "x", "diag".
      If a variable is missing, "data" will be None.
    
    Inputs:
      S           : happi S object
      frame       : index into time_steps
      time_steps  : list/array of diagnostic timesteps
      var_list    : list of variable names (keys from PARTICLE_MAP)
                    If None, loads all variables in PARTICLE_MAP.
                    
    Returns:
      results: dict
        results[name] = {"data": numpy array or None, "x": x-axis or None, "diag": ParticleBinning object or None}
    """
    if var_list is None:
        var_list = list(PARTICLE_MAP.keys())
    
    results = {}
    ts = time_steps[frame]
    
    for name in var_list:
        if name not in PARTICLE_MAP:
            print(f"Warning: variable '{name}' not in PARTICLE_MAP. Skipping.")
            results[name] = {"data": None, "x": None, "diag": None}
            continue
        
        idx = PARTICLE_MAP[name]
        try:
            diag = S.ParticleBinning(idx, timesteps=ts, average={"y":"all"})
            if diag is None:
                print(f"Warning: Diagnostic '{name}' at timestep {ts} returned None.")
                results[name] = {"data": None, "x": None, "diag": None}
                continue
    
            raw = np.array(diag.getData())
            if raw.size == 0:
                print(f"Warning: Diagnostic '{name}' at timestep {ts} returned empty data.")
                results[name] = {"data": None, "x": None, "diag": diag}
                continue
    
            data = np.squeeze(raw)
            # x-axis
            x_axis, _ = _try_get_axes(diag)
            if x_axis is None:
                if data.ndim == 1:
                    x_axis = np.arange(data.size)
                elif data.ndim == 2:
                    x_axis = np.arange(data.shape[0])
                else:
                    x_axis = None
    
            results[name] = {"data": data, "x": x_axis, "diag": diag}
    
        except Exception as e:
            print(f"Error loading diagnostic '{name}' at timestep {ts}: {e}")
            results[name] = {"data": None, "x": None, "diag": None}
    
    return results


#### for computing pressure and inverse beta

def compute_pressure_and_beta(S, frame, ion_mass, time_step, bound=1e-2):
        """
        Compute total, electron, and ion pressures, magnetic energy density, 
        and inverse plasma beta for a given simulation frame.

        Parameters
        ----------
        S : Smilei simulation object
        frame : int
                Index of the frame (e.g., 0, 1, 2, ...)
        time_step : list or array
                Array of simulation timesteps
        bound : float
                Minimum allowed pressure value to avoid division by zero (default=1e-1)

        Returns
        -------
        x_range : np.ndarray
                Centered x-coordinate array
        total_pressure : np.ndarray
                Total (electron + ion) pressure
        e_pressure : np.ndarray
                Electron pressure
        i_pressure : np.ndarray
                Ion pressure
        B_data_yav_sq : np.ndarray
                Magnetic energy density (Bx^2 + By^2 + Bz^2)
        inverse_beta_1d_y_av : np.ndarray
                (B^2/2) / total_pressure, with NaN where pressure < bound
        """

        # ---------- Variable lists ----------
        pressure_vars = ["e_pxx", "e_pyy", "e_pzz", "i_pxx", "i_pyy", "i_pzz"]
        current_vars = ["e_jx", "e_jy", "e_jz", "i_jx", "i_jy", "i_jz"]
        density_vars = ["e_n", "i_n"]

        # ---------- Read data ----------
        pressure_1d = particle_1d_yavg_safe(S, frame, time_step, var_list=pressure_vars)
        current_1d  = particle_1d_yavg_safe(S, frame, time_step, var_list=current_vars)
        density_1d  = particle_1d_yavg_safe(S, frame, time_step, var_list=density_vars)

        # ---------- Densities ----------
        e_n, i_n = density_1d["e_n"]["data"], density_1d["i_n"]["data"]

        # ---------- Currents ----------
        e_jx, e_jy, e_jz = current_1d["e_jx"]["data"], current_1d["e_jy"]["data"], current_1d["e_jz"]["data"]
        i_jx, i_jy, i_jz = current_1d["i_jx"]["data"], current_1d["i_jy"]["data"], current_1d["i_jz"]["data"]

        # ---------- Drift velocities ----------
        e_vx, e_vy, e_vz = -e_jx / e_n, -e_jy / e_n, -e_jz / e_n
        i_vx, i_vy, i_vz =  i_jx / i_n,  i_jy / i_n,  i_jz / i_n

        # ---------- Raw pressures ----------
        e_pxx_raw, e_pyy_raw, e_pzz_raw = pressure_1d["e_pxx"]["data"], pressure_1d["e_pyy"]["data"], pressure_1d["e_pzz"]["data"]
        i_pxx_raw, i_pyy_raw, i_pzz_raw = pressure_1d["i_pxx"]["data"], pressure_1d["i_pyy"]["data"], pressure_1d["i_pzz"]["data"]

        # ---------- Thermal pressures (subtract bulk flow) ----------
        e_pxx, e_pyy, e_pzz = e_pxx_raw - e_n*e_vx**2, e_pyy_raw - e_n*e_vy**2, e_pzz_raw - e_n*e_vz**2
        i_pxx, i_pyy, i_pzz = i_pxx_raw - ion_mass*i_n*i_vx**2, i_pyy_raw - ion_mass*i_n*i_vy**2, i_pzz_raw - ion_mass*i_n*i_vz**2

        # ---------- Scalar pressures ----------
        e_pressure = (1.0/3.0) * (e_pxx + e_pyy + e_pzz)
        i_pressure = (1.0/3.0) * (i_pxx + i_pyy + i_pzz)
        total_pressure = e_pressure + i_pressure

        # ---------- Magnetic fields ----------
        Bx1_yav = S.Field(0, 'Bx', average={"y": "all"})
        By1_yav = S.Field(0, 'By', average={"y": "all"})
        Bz1_yav = S.Field(0, 'Bz', average={"y": "all"})

        Bx_data_yav = np.squeeze(np.array(Bx1_yav.getData(timestep=time_step[frame])), axis=0)
        By_data_yav = np.squeeze(np.array(By1_yav.getData(timestep=time_step[frame])), axis=0)
        Bz_data_yav = np.squeeze(np.array(Bz1_yav.getData(timestep=time_step[frame])), axis=0)

        # ---------- Magnetic energy density ----------
        B_data_yav_sq = Bx_data_yav**2 + By_data_yav**2 + Bz_data_yav**2

        # ---------- Inverse beta ----------
        mask = total_pressure > bound
        inverse_beta_1d_y_av = np.full_like(total_pressure, np.nan)
        inverse_beta_1d_y_av[mask] = (B_data_yav_sq[mask] / 2.0) / total_pressure[mask]

        # ---------- Centered x-axis ----------
        x_Bz1 = Bz1_yav.getAxis('x')
        # xc_offset = -0.5 * (x_Bz1[0] + x_Bz1[-1])
        # x_range = x_Bz1 + xc_offset

        x_range = x_Bz1
    
        return x_range, total_pressure, e_pressure, i_pressure, B_data_yav_sq, inverse_beta_1d_y_av    

###############################


def compute_j_v_species_1d(S, frame, ion_mass, time_step, bound=1e-2):

        # ---------- Variable lists ----------
        pressure_vars = ["e_pxx", "e_pyy", "e_pzz", "i_pxx", "i_pyy", "i_pzz"]
        current_vars = ["e_jx", "e_jy", "e_jz", "i_jx", "i_jy", "i_jz"]
        density_vars = ["e_n", "i_n"]

        # ---------- Read data ----------
        pressure_1d = particle_1d_yavg_safe(S, frame, time_step, var_list=pressure_vars)
        current_1d  = particle_1d_yavg_safe(S, frame, time_step, var_list=current_vars)
        density_1d  = particle_1d_yavg_safe(S, frame, time_step, var_list=density_vars)

        # ---------- Densities ----------
        e_n, i_n = density_1d["e_n"]["data"], density_1d["i_n"]["data"]

        # ---------- Currents ----------
        e_jx, e_jy, e_jz = current_1d["e_jx"]["data"], current_1d["e_jy"]["data"], current_1d["e_jz"]["data"]
        i_jx, i_jy, i_jz = current_1d["i_jx"]["data"], current_1d["i_jy"]["data"], current_1d["i_jz"]["data"]

        # ---------- Drift velocities ----------
        e_vx, e_vy, e_vz = -e_jx / e_n, -e_jy / e_n, -e_jz / e_n
        i_vx, i_vy, i_vz =  i_jx / i_n,  i_jy / i_n,  i_jz / i_n

        Bz1_yav = S.Field(0, 'Bz', average={"y": "all"})
        x_Bz1 = Bz1_yav.getAxis('x')
        x_range = x_Bz1
    
        return x_range, e_n, i_n, e_jx, e_jy, e_jz, i_jx, i_jy, i_jz
##########################################################################################
def trim_right_vacuum_region(x, total_pressure, Bsq=None, n_trim_extra=5):
    """
    Automatically trim arrays where total_pressure becomes non-positive near the right boundary.
   
    Parameters
    ----------
    x : ndarray
        1D spatial coordinate array (e.g. x_range)
    total_pressure : ndarray
        Total (thermal) pressure array
    Bsq : ndarray or None
        Optional B^2 array to trim similarly
    n_trim_extra : int
        Number of extra cells to remove to the left of the zero-crossing (safety margin)

    Returns
    -------
    x_valid, P_valid, Bsq_valid, trim_index
    """
    # make copies
    x = np.asarray(x)
    P = np.asarray(total_pressure)

    # find the first index (from right) where pressure crosses <= 0
    idx_cross = None
    for i in range(len(P)-1, 0, -1):
        if (P[i] <= 0 and P[i-1] > 0) or (P[i] < 0):
            idx_cross = i
            break

    if idx_cross is None:
        # No crossing found; return full arrays
        print("No negative or zero pressure found — returning full arrays.")
        if Bsq is not None:
            return x, P, Bsq, len(P)
        else:
            return x, P, None, len(P)

    # Apply safety margin
    trim_index = max(idx_cross - n_trim_extra, 0)
    if trim_index == 0:
        print("Warning: trimming would remove entire array; check data.")

    # Trim arrays
    x_valid = x[:trim_index]
    P_valid = P[:trim_index]
    Bsq_valid = None
    if Bsq is not None:
        Bsq_valid = Bsq[:trim_index]

    print(f"Trimming arrays at index {trim_index} (x = {x[trim_index]:.3f}).")
    return x_valid, P_valid, Bsq_valid, trim_index


######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
#### The follwoing code is for fine tuning the pressure terms. 

# --------------------------
# 1) Trim low/negative pressure region near right boundary
# --------------------------
def find_trim_index_right(total_pressure, tol_frac=1e-8, min_negative_run=1):
              """
              Find first index (from left) where total_pressure <= tol (near right boundary)
              but robust to small isolated dips:
                     - tol_frac: threshold relative to max(total_pressure) (absolute tol = tol_frac * max)
                     - min_negative_run: require at least this many contiguous <= tol points to accept trimming
              Returns: trim_index (int) such that valid region = [:trim_index]
                                           If no trimming needed returns len(total_pressure)
              """
              P = np.asarray(total_pressure)
              if P.size == 0:
                            return 0
              tol = tol_frac * max(np.nanmax(P), 1.0)  # avoid zero tol if P very small
              # boolean array of "bad" cells (<= tol)
              bad = (P <= tol) | np.isnan(P)
              N = len(P)

              # search from left to right for first starting position of a run of length >= min_negative_run that continues to end or at least exists
              for i in range(0, N):
                            # only consider if this index is bad
                            if not bad[i]:
                                          continue
                            # check contiguous run length from i
                            run_len = 0
                            j = i
                            while j < N and bad[j]:
                                          run_len += 1
                                          j += 1
                            if run_len >= min_negative_run:
                                          # i is the first index where we should cut everything to the right of (i - safety_margin)
                                          return i
                            else:
                                          # skip past this short run
                                          i = j
              # no robust bad region found
              return N

def trim_right_region_with_margin(x, total_pressure, Bsq=None, n_trim_extra=5, tol_frac=1e-8, min_negative_run=1):
              """
              Trims arrays to exclude zone where total_pressure becomes <= tol.
              Returns x_valid, P_valid, Bsq_valid, trim_index
              """
              trim_idx = find_trim_index_right(total_pressure, tol_frac=tol_frac, min_negative_run=min_negative_run)
              # apply safety margin (move left by n_trim_extra)
              trim_idx = max(trim_idx - n_trim_extra, 0)
              x_valid = x[:trim_idx].copy()
              P_valid = total_pressure[:trim_idx].copy()
              Bsq_valid = None
              if Bsq is not None:
                            Bsq_valid = Bsq[:trim_idx].copy()
              return x_valid, P_valid, Bsq_valid, trim_idx

# --------------------------
# 2) inverse beta safe compute + optional spike handling
# --------------------------
def compute_inverse_beta(Bsq, Ptot, method='mask', pressure_floor_frac=1e-6,
                        smoothing_sigma=0, median_filter_size=None, clip_percentile=99.5):
              """
              Compute inverse_beta safely.
              method: 'mask' or 'floor' or 'floor_and_clip'
                     - 'mask': return inverse_beta with NaNs where Ptot <= floor
                     - 'floor': floor pressure at pressure_floor_frac*max(P), return full array
                     - 'floor_and_clip': same as floor, then clip very large values to clip_percentile for visualization
              smoothing_sigma: apply gaussian smoothing to both Bsq/2 and Ptot before division (after floor or mask)
              median_filter_size: if int>1, apply median filter to inverse_beta to remove isolated spikes
              clip_percentile: percentile to cap the values (only for floor_and_clip)
              Returns inverse_beta (np.ndarray), mask_valid (bool mask where phys data)
              """
              Bsq = np.asarray(Bsq)
              Ptot = np.asarray(Ptot)
              Pfloor = pressure_floor_frac * max(np.nanmax(Ptot), 1.0)

              if method == 'mask':
                            mask = Ptot > Pfloor
                            inv = np.full_like(Ptot, np.nan, dtype=float)
                            if smoothing_sigma and smoothing_sigma > 0:
                                          Bv = gaussian_filter1d(Bsq/2.0, sigma=smoothing_sigma)
                                          Pv = gaussian_filter1d(Ptot, sigma=smoothing_sigma)
                            else:
                                          Bv = Bsq/2.0
                                          Pv = Ptot
                            inv[mask] = Bv[mask] / Pv[mask]
              else:
                            # floor method
                            Pv_safe = np.maximum(Ptot, Pfloor)
                            if smoothing_sigma and smoothing_sigma > 0:
                                          Bv = gaussian_filter1d(Bsq/2.0, sigma=smoothing_sigma)
                                          Pv_safe = gaussian_filter1d(Pv_safe, sigma=smoothing_sigma)
                            else:
                                          Bv = Bsq/2.0
                            inv = Bv / Pv_safe
                            mask = Pv_safe > Pfloor

                            if method == 'floor_and_clip':
                                          # cap very large values at a percentile for visualization
                                          if np.isfinite(clip_percentile) and clip_percentile < 100.0:
                                                        cap = np.nanpercentile(inv[mask], clip_percentile)
                                                        inv = np.minimum(inv, cap)

              # optional median filter to remove isolated spikes
              if median_filter_size and median_filter_size > 1:
                            inv = median_filter(inv, size=median_filter_size)

              return inv, mask

# --------------------------
# 3) Fitting the inverse_beta transition region automatically
# --------------------------
def user_model_inverse_beta(x, A, w, R, alpha, x0):
              """
              Model:
                     inv_fit = A * ( 0.5*(tanh^2((x-x0)/w) + R)**2 ) / ( 1 - alpha*tanh((x-x0)/w) - alpha*tanh^2((x-x0)/w) )
              """
              t = np.tanh((x - x0) / w)
              num = 0.5 * (t**2 + R)**2
              den = 1.0 - alpha * t - alpha * t**2
              # avoid divide by zero in model evaluation by small epsilon
              den = np.where(np.abs(den) < 1e-12, np.sign(den) * 1e-12, den)
              return A * (num / den)

def find_contiguous_regions_above_threshold(x, invbeta, threshold=5.0, min_len=10):
              """
              Returns list of (start_idx, end_idx) contiguous index ranges where invbeta >= threshold
              and length >= min_len.
              """
              mask = np.isfinite(invbeta) & (invbeta >= threshold)
              regions = []
              N = len(invbeta)
              i = 0
              while i < N:
                            if not mask[i]:
                                          i += 1
                                          continue
                            j = i
                            while j < N and mask[j]:
                                          j += 1
                            if (j - i) >= min_len:
                                          regions.append((i, j))
                            i = j
              return regions

def fit_inverse_beta_transition(x, invbeta, threshold=5.0, min_len=20,
                                                                                                                p0=None, bounds=None, maxfev=5000):
              """
              Automatically select region(s) where invbeta >= threshold and length >= min_len,
              pick the longest region, and fit the user_model_inverse_beta.
              Returns: popt, pcov, fit_x, fit_y, region_slice (start,end)
                                           and computed beta_left, beta_right and delta_beta (beta_right-beta_left)
              """
              x = np.asarray(x)
              inv = np.asarray(invbeta)

              regions = find_contiguous_regions_above_threshold(x, inv, threshold=threshold, min_len=min_len)
              if not regions:
                            raise RuntimeError("No fitting region found with threshold/min_len.")

              # pick the longest region to avoid small isolated patches
              region = max(regions, key=lambda s: s[1] - s[0])
              start, end = region
              x_fit = x[start:end]
              y_fit = inv[start:end]

              # default initial guess (if none provided)
              if p0 is None:
                            A0 = np.nanmedian(y_fit)
                            w0 = max(1.0, (x_fit[-1] - x_fit[0]) / 6.0)        # roughly
                            R0 = 0.0
                            alpha0 = 0.1
                            x00 = np.mean(x_fit)
                            p0 = [A0, w0, R0, alpha0, x00]

              if bounds is None:
                            # set fairly broad bounds to allow robust convergence but avoid unphysical negatives
                            bounds = (
                                          [0.0, 1e-6, -5.0, -5.0, x_fit[0]],       # lower bounds
                                          [np.inf, (x_fit[-1]-x_fit[0])*5, 5.0, 5.0, x_fit[-1]]  # upper
                            )

              # fit
              try:
                            popt, pcov = curve_fit(user_model_inverse_beta, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=maxfev)
              except Exception as e:
                            raise RuntimeError(f"Fit failed: {e}")

              # Evaluate fit on region endpoints to get asymptotic-ish left/right inverse_beta
              inv_left_fit = user_model_inverse_beta(x_fit[0], *popt)
              inv_right_fit = user_model_inverse_beta(x_fit[-1], *popt)
              # compute beta = 1 / invbeta; avoid divide by zero
              eps = 1e-12
              beta_left = 1.0 / max(inv_left_fit, eps)
              beta_right = 1.0 / max(inv_right_fit, eps)
              delta_beta = beta_right - beta_left

              return {
                            'popt': popt, 'pcov': pcov,
                            'x_fit': x_fit, 'y_fit': y_fit, 'region': (start, end),
                            'inv_left_fit': inv_left_fit, 'inv_right_fit': inv_right_fit,
                            'beta_left': beta_left, 'beta_right': beta_right, 'delta_beta': delta_beta
              }

################################################################################################################
#####      ----------------------    Fitting for inverse beta     ------------------------       ###############
################################################################################################################



# --- user model (as you specified)
def inv_model(x, A, w, R, alpha, x0):
    t = np.tanh((x - x0) / w)
    num = 0.5 * (t**2 + R)**2
    den = 1.0 - alpha * t - alpha * t**2
    den = np.where(np.abs(den) < 1e-12, np.sign(den) * 1e-12, den)
    return A * (num / den)

# --- helper: despike & smooth
def preprocess_inv(inv, median_size=3, gauss_sigma=1.0, despike_percentile=99.5, despike_replace='interp'):
    """
    inv : 1D array (inverse_beta)
    median_size : int, median filter window to remove isolated spikes
    gauss_sigma : float, gaussian smoothing sigma (low-pass)
    despike_percentile: values above this percentile are considered spikes
    despike_replace : 'interp' or 'clip' - how to handle spikes
    returns: processed_inv
    """
    inv = np.asarray(inv, dtype=float).copy()
    # 1) median filter to remove single-cell spikes
    if median_size is not None and median_size > 1:
        inv_med = median_filter(inv, size=median_size)
    else:
        inv_med = inv.copy()

    # 2) gaussian smooth lightly
    if gauss_sigma and gauss_sigma > 0:
        inv_smooth = gaussian_filter1d(inv_med, sigma=gauss_sigma)
    else:
        inv_smooth = inv_med.copy()

    # 3) despike: identify very large spikes relative to distribution
    if despike_percentile is not None and 0 < despike_percentile < 100:
        threshold = np.nanpercentile(inv_smooth[np.isfinite(inv_smooth)], despike_percentile)
        spikes = (inv_smooth > threshold) | ~np.isfinite(inv_smooth)
        if np.any(spikes):
            if despike_replace == 'interp':
                # simple 1D linear interpolation to fill spike locations
                good_idx = np.where(~spikes)[0]
                bad_idx = np.where(spikes)[0]
                if good_idx.size >= 2:
                    interp = np.interp(bad_idx, good_idx, inv_smooth[good_idx])
                    inv_smooth[bad_idx] = interp
                else:
                    # cannot interp; clip
                    inv_smooth[spikes] = threshold
            else:
                inv_smooth[spikes] = threshold
    return inv_smooth

# --- helper: select best contiguous region
# def pick_fitting_region(x, inv, threshold=5.0, min_len=20, max_gap=0):
#     """
#     Return (start_idx, end_idx) of region to fit.
#     Strategy:
#       - find mask inv>=threshold (finite)
#       - extract contiguous segments
#       - optionally allow small gaps (max_gap) to be bridged
#       - require length >= min_len
#       - choose longest valid segment
#     """
#     inv = np.asarray(inv)
#     mask = np.isfinite(inv) & (inv >= threshold)
#     if not np.any(mask):
#         return None
#     # find contiguous runs
#     runs = []
#     N = len(mask)
#     i = 0
#     while i < N:
#         if not mask[i]:
#             i += 1
#             continue
#         j = i
#         while j < N and mask[j]:
#             j += 1
#         runs.append((i, j))
#         i = j
#     # optionally, try to merge runs separated by small gaps (max_gap>0)
#     if max_gap > 0 and len(runs) > 1:
#         merged = []
#         cur_s, cur_e = runs[0]
#         for s,e in runs[1:]:
#             gap = s - cur_e
#             if gap <= max_gap:
#                 # merge
#                 cur_e = e
#             else:
#                 merged.append((cur_s, cur_e))
#                 cur_s, cur_e = s, e
#         merged.append((cur_s, cur_e))
#         runs = merged
#     # keep only runs >= min_len
#     runs = [r for r in runs if (r[1]-r[0]) >= min_len]
#     if not runs:
#         return None
#     # pick the longest
#     run = max(runs, key=lambda r: r[1]-r[0])
#     return run


def pick_fitting_region(
    x, inv,
    threshold=5.0, min_len=20, max_gap=0,
    right_exclude_frac=0.02,   # exclude last 2% of domain
    outlier_factor=5.0,        # discard points > mean + 5×std
    verbose=False
):
    """
    Improved region selector that avoids boundary spikes and extreme outliers.
    
    Parameters
    ----------
    x : array
        spatial array
    inv : array
        inverse beta array
    threshold : float
        minimum inverse-beta to be considered part of the transition region
    min_len : int
        minimum length of region to accept
    max_gap : int
        allowed small gap to merge runs
    right_exclude_frac : float
        fraction of the rightmost domain to exclude (default 0.02 = 2%)
    outlier_factor : float
        reject points > mean + outlier_factor * std
    verbose : bool
        print diagnostic info
    """
    inv = np.asarray(inv, dtype=float)
    x = np.asarray(x, dtype=float)

    # --- Step 1: mask finite points
    mask = np.isfinite(inv)
    inv_valid = inv[mask]
    x_valid = x[mask]
    if inv_valid.size == 0:
        return None

    # --- Step 2: exclude the rightmost region (vacuum / boundary)
    x_cut = x_valid[-1] - right_exclude_frac * (x_valid[-1] - x_valid[0])
    mask &= x <= x_cut

    # --- Step 3: remove statistical outliers (very large spikes)
    inv_mean = np.nanmean(inv[mask])
    inv_std = np.nanstd(inv[mask])
    spike_mask = inv < (inv_mean + outlier_factor * inv_std)
    mask &= spike_mask

    # --- Step 4: apply threshold
    mask &= inv >= threshold

    # --- Step 5: extract contiguous runs
    runs = []
    N = len(mask)
    i = 0
    while i < N:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < N and mask[j]:
            j += 1
        runs.append((i, j))
        i = j

    # --- Step 6: optionally merge short gaps
    if max_gap > 0 and len(runs) > 1:
        merged = []
        cur_s, cur_e = runs[0]
        for s, e in runs[1:]:
            if s - cur_e <= max_gap:
                cur_e = e
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        runs = merged

    # --- Step 7: keep long enough runs
    runs = [r for r in runs if (r[1] - r[0]) >= min_len]
    if not runs:
        if verbose:
            print("⚠️ No valid contiguous region found after filtering.")
        return None

    # --- Step 8: pick the *last* long region before the right boundary
    # (this avoids fitting early features)
    run = runs[-1]
    if verbose:
        s, e = run
        print(f"✅ Fitting region: x={x[s]:.2f} → {x[e-1]:.2f} "
              f"({e-s} points, mean={inv_mean:.2f}, std={inv_std:.2f})")
    return run


# --- robust fit using least_squares with soft L1 or Huber
def robust_fit_inv(x_fit, y_fit, p0=None, bounds=None, loss='soft_l1', f_scale=1.0, max_nfev=5000):
    """
    Fit inv_model using robust least squares.
    loss: 'soft_l1' or 'huber' or 'linear'
    f_scale: parameter controlling transition for soft_l1/huber
    """
    if p0 is None:
        # fallback initial guesses
        A0 = np.nanmedian(y_fit)
        w0 = max(1.0, (x_fit[-1]-x_fit[0]) / 6.0)
        R0 = 0.0
        a0 = 0.1
        x00 = np.mean(x_fit)
        p0 = [A0, w0, R0, a0, x00]
    p0 = np.array(p0, dtype=float)

    if bounds is None:
        # assign wide bounds but avoid division by zero in w
        bounds = (
            [0.0, 1e-6, -10.0, -5.0, x_fit[0]],
            [np.inf, (x_fit[-1]-x_fit[0])*5.0, 10.0, 5.0, x_fit[-1]]
        )

    def resid(p, x, y):
        return inv_model(x, *p) - y

    res = least_squares(resid, p0, args=(x_fit, y_fit), bounds=bounds,
                        loss=loss, f_scale=f_scale, max_nfev=max_nfev)
    return res
'''
# --- top-level routine: preprocess -> pick region -> robust fit -> compute delta_beta
def fit_inverse_beta_pipeline(x, inv_raw,
                              preprocess_kwargs=None,
                              pick_kwargs=None,
                              fit_kwargs=None,
                              compute_at_offset=3.0):
    """
    x : 1D x array
    inv_raw : raw inverse_beta 1D array
    preprocess_kwargs : dict for preprocess_inv
    pick_kwargs : dict for pick_fitting_region
    fit_kwargs : dict for robust_fit_inv
    compute_at_offset : multiplier of fitted w to pick left/right evaluation points (e.g. x0 +/- 3*w)
    Returns dict with fit result, x_fit,y_fit, and computed beta_left/right/delta
    """
    if preprocess_kwargs is None:
        preprocess_kwargs = dict(median_size=3, gauss_sigma=1.0, despike_percentile=99.5, despike_replace='interp')
    if pick_kwargs is None:
        pick_kwargs = dict(threshold=5.0, min_len=20, max_gap=2)
    if fit_kwargs is None:
        fit_kwargs = dict(p0=None, bounds=None, loss='soft_l1', f_scale=1.0, max_nfev=5000)

    inv_proc = preprocess_inv(inv_raw, **preprocess_kwargs)

    region = pick_fitting_region(x, inv_proc, **pick_kwargs)
    if region is None:
        raise RuntimeError("No valid fitting region found after preprocessing.")
    s,e = region
    x_fit = x[s:e]
    y_fit = inv_proc[s:e]

    # do robust fit
    res = robust_fit_inv(x_fit, y_fit, **fit_kwargs)
    p = res.x
    # fitted parameters
    A, w, R, alpha, x0 = p

    # choose evaluation points away from edges: x0 +/- compute_at_offset * w
    x_left_eval = x0 - compute_at_offset * w
    x_right_eval = x0 + compute_at_offset * w

    inv_left = inv_model(x_left_eval, *p)
    inv_right = inv_model(x_right_eval, *p)
    # convert inv -> beta safely
    eps = 1e-12
    beta_left = 1.0 / max(inv_left, eps)
    beta_right = 1.0 / max(inv_right, eps)
    delta_beta = beta_right - beta_left

    return {
        'res': res,
        'popt': p,
        'x_fit': x_fit, 'y_fit': y_fit,
        'region': region,
        'inv_left': inv_left, 'inv_right': inv_right,
        'beta_left': beta_left, 'beta_right': beta_right,
        'delta_beta': delta_beta,
        'inv_proc': inv_proc
    }
'''


def fit_inverse_beta_pipeline_improved(
    x, inv_raw,
    median_size=5, gauss_sigma=1.2, despike_percentile=99.0,
    threshold=0.5, grad_threshold=0.015,
    min_len=40, max_gap=2,
    extend_left=25, extend_right=0,
    right_limit_frac=0.98,
    loss='huber', f_scale=0.5,
    compute_at_offset=3.0, verbose=True):
    """
    Improved version: combines threshold+gradient selection and allows region extension.
    """

    # --- Preprocess
    inv = inv_raw.copy()
    inv = median_filter(inv, size=median_size)
    inv = gaussian_filter1d(inv, sigma=gauss_sigma)
    cap = np.nanpercentile(inv[np.isfinite(inv)], despike_percentile)
    inv = np.clip(inv, 0, cap)

    # --- Combined mask: value and gradient
    grad = np.gradient(inv, x)
    mask = (inv >= threshold) | (np.abs(grad) >= grad_threshold)
    if not np.any(mask):
        raise RuntimeError("No valid region detected (mask empty).")

    # --- Find contiguous segments
    regions = []
    i = 0
    N = len(mask)
    while i < N:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < N and mask[j]:
            j += 1
        if j - i >= min_len:
            regions.append((i, j))
        i = j
    if not regions:
        raise RuntimeError("No region found passing min_len filter.")

    # --- Choose the longest region
    s, e = max(regions, key=lambda r: r[1]-r[0])

    # --- Extend to left/right
    s = max(0, s - extend_left)
    e = min(int(right_limit_frac * N), e + extend_right)

    x_fit = x[s:e]
    y_fit = inv[s:e]
    if verbose:
        print(f"Fitting region: {x_fit[0]:.2f} → {x_fit[-1]:.2f} "
              f"({e-s} pts, extended {extend_left} left, limited right)")

    # --- Robust least-squares fit
    A0 = np.nanmedian(y_fit)
    w0 = max(1.0, (x_fit[-1] - x_fit[0]) / 6.0)
    R0, a0, x00 = 0.0, 0.1, np.mean(x_fit)
    p0 = [A0, w0, R0, a0, x00]
    bounds = ([0, 1e-6, -10, -5, x_fit[0]],
              [np.inf, (x_fit[-1]-x_fit[0])*5, 10, 5, x_fit[-1]])

    def resid(p, x, y): return inv_model(x, *p) - y
    res = least_squares(resid, p0, args=(x_fit, y_fit),
                        bounds=bounds, loss=loss, f_scale=f_scale,
                        max_nfev=8000)

    A, w, R, alpha, x0 = res.x
    xL, xR = x0 - compute_at_offset*w, x0 + compute_at_offset*w
    invL, invR = inv_model(xL, *res.x), inv_model(xR, *res.x)
    eps = 1e-12
    betaL, betaR = 1.0 / max(invL, eps), 1.0 / max(invR, eps)
    delta_beta = betaR - betaL

    return dict(
        res=res, popt=res.x, x_fit=x_fit, y_fit=y_fit,
        region=(s, e), inv_proc=inv,
        inv_left=invL, inv_right=invR,
        beta_left=betaL, beta_right=betaR,
        delta_beta=delta_beta)


# def fit_inverse_beta_pipeline_cleaned(
#     x, inv_raw,
#     median_size=5, gauss_sigma=1.2,
#     despike_percentile=99.0,
#     threshold=0.5, grad_threshold=0.015,
#     min_len=40, extend_left=25,
#     right_limit_frac=0.98,
#     loss='huber', f_scale=0.5,
#     compute_at_offset=3.0,
#     right_outlier_frac=0.05,
#     outlier_sigma_factor=5.0,
#     verbose=True):
#     """
#     Improved inverse-beta fitting:
#     - Smooths and clips data
#     - Removes high-value outliers near right boundary
#     - Fits tanh-based model robustly
#     """

#     inv = inv_raw.copy()
#     inv = median_filter(inv, size=median_size)
#     inv = gaussian_filter1d(inv, sigma=gauss_sigma)

#     # --- Optional clipping for extreme spikes
#     cap = np.nanpercentile(inv[np.isfinite(inv)], despike_percentile)
#     inv = np.clip(inv, 0, cap)

#     # --- Identify and discard right-boundary outliers
#     N = len(inv)
#     right_start = int((1.0 - right_outlier_frac) * N)
#     right_vals = inv[right_start:]
#     mean_val = np.nanmean(inv[:right_start])
#     mask_good = np.ones_like(inv, dtype=bool)
#     mask_good[right_start:] = right_vals < outlier_sigma_factor * mean_val

#     if verbose and np.sum(~mask_good[right_start:]) > 0:
#         print(f"Discarded {np.sum(~mask_good[right_start:])} outlier points near right boundary.")

#     # Keep only valid region
#     x_clean = x[mask_good]
#     inv_clean = inv[mask_good]

#     # --- Combined mask (value and gradient) to find fit region
#     grad = np.gradient(inv_clean, x_clean)
#     mask = (inv_clean >= threshold) | (np.abs(grad) >= grad_threshold)

#     # Find contiguous valid region(s)
#     regions = []
#     i = 0
#     N = len(mask)
#     while i < N:
#         if not mask[i]:
#             i += 1
#             continue
#         j = i
#         while j < N and mask[j]:
#             j += 1
#         if j - i >= min_len:
#             regions.append((i, j))
#         i = j

#     if not regions:
#         raise RuntimeError("No valid fitting region found.")
#     s, e = max(regions, key=lambda r: r[1] - r[0])
#     s = max(0, s - extend_left)
#     e = min(int(right_limit_frac * N), e)
#     x_fit, y_fit = x_clean[s:e], inv_clean[s:e]

#     # --- Robust fit
#     A0 = np.nanmedian(y_fit)
#     w0 = max(1.0, (x_fit[-1] - x_fit[0]) / 6.0)
#     R0, a0, x00 = 0.0, 0.1, np.mean(x_fit)
#     p0 = [A0, w0, R0, a0, x00]
#     bounds = ([0, 1e-6, -10, -5, x_fit[0]],
#               [np.inf, (x_fit[-1]-x_fit[0])*5, 10, 5, x_fit[-1]])

#     def resid(p, x, y): return inv_model(x, *p) - y
#     res = least_squares(resid, p0, args=(x_fit, y_fit),
#                         bounds=bounds, loss=loss, f_scale=f_scale, max_nfev=8000)

#     # --- Compute delta_beta
#     A, w, R, alpha, x0 = res.x
#     xL, xR = x0 - compute_at_offset*w, x0 + compute_at_offset*w
#     invL, invR = inv_model(xL, *res.x), inv_model(xR, *res.x)
#     eps = 1e-12
#     betaL, betaR = 1.0 / max(invL, eps), 1.0 / max(invR, eps)
#     delta_beta = betaR - betaL

#     return dict(
#         res=res, popt=res.x, x_fit=x_fit, y_fit=y_fit,
#         region=(s, e), inv_proc=inv, inv_clean=inv_clean,
#         inv_left=invL, inv_right=invR,
#         beta_left=betaL, beta_right=betaR,
#         delta_beta=delta_beta
#     )



def fit_inverse_beta_pipeline_cleaned(
    x, inv_raw,
    median_size=5, gauss_sigma=1.2,
    despike_percentile=99.0,
    threshold=0.5, grad_threshold=0.015,
    min_len=40, extend_left=25,
    right_limit_frac=0.98,
    loss='huber', f_scale=0.5,
    compute_at_offset=3.0,
    right_outlier_frac=0.05,
    outlier_sigma_factor=5.0,
    verbose=True):
    """
    Improved inverse-beta fitting:
      - Smooth + median-filter
      - Clip extreme percentiles
      - Remove extreme right-boundary outliers
      - Build mask on the cleaned arrays
      - Pick contiguous region, then *expand/limit* indices using extend_left and right_limit_frac
      - Fit using robust least_squares
    Returns a dict (same keys as before) plus diagnostic info.
    """

    # --- 1) basic preprocessing on inv
    inv = np.asarray(inv_raw, dtype=float).copy()
    if median_size is not None and median_size > 1:
        inv = median_filter(inv, size=median_size)
    if gauss_sigma and gauss_sigma > 0:
        inv = gaussian_filter1d(inv, sigma=gauss_sigma)

    # Clip extreme large spikes at high percentile (keeps shape but removes extreme top outliers)
    finite_mask = np.isfinite(inv)
    if np.any(finite_mask):
        cap = np.nanpercentile(inv[finite_mask], despike_percentile)
        inv = np.clip(inv, None, cap)
    else:
        raise RuntimeError("Input inv is all non-finite.")

    # --- 2) remove right-boundary outliers (operate on indices)
    N = len(inv)
    right_start_idx = int(max(0, np.floor((1.0 - right_outlier_frac) * N)))
    # Compute mean excluding the right-outlier window
    if right_start_idx > 1:
        mean_val = np.nanmean(inv[:right_start_idx])
        std_val = np.nanstd(inv[:right_start_idx])
    else:
        mean_val = np.nanmean(inv)
        std_val = np.nanstd(inv)

    mask_outliers = np.ones(N, dtype=bool)
    if right_start_idx < N:
        # mark as bad those right-side points which are >> mean
        bad_right = (inv[right_start_idx:] > outlier_sigma_factor * max(mean_val, 1e-12))
        mask_outliers[right_start_idx:] = ~bad_right  # True keep if NOT bad
        if verbose:
            nbad = np.sum(bad_right)
            if nbad:
                print(f"[preproc] removed {nbad} high-value outlier(s) in right {right_outlier_frac*100:.2f}% window")

    # Build cleaned arrays (apply outlier mask and finite mask)
    keep_mask = mask_outliers & finite_mask
    x_clean = np.asarray(x)[keep_mask]
    inv_clean = inv[keep_mask]
    if x_clean.size < min_len:
        raise RuntimeError("Not enough cleaned points remain after outlier removal.")

    # --- 3) build selection mask on cleaned arrays (value or gradient)
    grad_clean = np.gradient(inv_clean, x_clean)
    mask_sel = (np.isfinite(inv_clean)) & ((inv_clean >= threshold) | (np.abs(grad_clean) >= grad_threshold))

    # find contiguous runs on the cleaned mask
    runs = []
    i = 0
    M = len(mask_sel)
    while i < M:
        if not mask_sel[i]:
            i += 1
            continue
        j = i
        while j < M and mask_sel[j]:
            j += 1
        if (j - i) >= min_len:
            runs.append((i, j))
        i = j

    if not runs:
        # If nothing found, relax criteria (use only grad) once before failing
        if verbose:
            print("[pick] no runs with current threshold/grad_threshold/min_len — trying relaxed criteria (lower threshold)")
        mask_sel_relaxed = (np.isfinite(inv_clean)) & (np.abs(grad_clean) >= (grad_threshold * 0.2))
        runs = []
        i = 0
        while i < M:
            if not mask_sel_relaxed[i]:
                i += 1
                continue
            j = i
            while j < M and mask_sel_relaxed[j]:
                j += 1
            if (j - i) >= max(5, min_len // 4):
                runs.append((i, j))
            i = j
        if not runs:
            raise RuntimeError("No valid fitting region found after relaxed attempt.")

    # Choose the longest run (or last run if you prefer last: runs[-1])
    # To prefer the right-most valid transition region, pick the run with largest end index
    run = max(runs, key=lambda r: r[1])   # pick the run that extends farthest to the right
    s_clean, e_clean = run  # indices into cleaned arrays

    # --- 4) Expand left (in cleaned-array indices) and clip right by right_limit_frac
    # convert right_limit_frac to index on cleaned array
    len_clean = len(x_clean)
    right_limit_idx = int(np.floor(right_limit_frac * len_clean))
    # apply extend_left (in points)
    s_expanded = max(0, s_clean - int(extend_left))
    e_expanded = min(len_clean, e_clean)  # keep e as found
    # enforce right_limit
    e_expanded = min(e_expanded, right_limit_idx)
    if e_expanded - s_expanded < min_len:
        # Try to enlarge to meet min_len by moving left if possible
        s_expanded = max(0, e_expanded - min_len)

    x_fit = x_clean[s_expanded:e_expanded]
    y_fit = inv_clean[s_expanded:e_expanded]
    if x_fit.size < 6:
        raise RuntimeError("Selected fit window too small after expansion/clipping.")

    if verbose:
        print(f"[region] cleaned_len={len_clean}, chosen_clean_run=({s_clean},{e_clean}), "
              f"expanded=({s_expanded},{e_expanded}), x_fit_range=({x_fit[0]:.3f},{x_fit[-1]:.3f}), nfit={x_fit.size}")

    # --- 5) Fit (robust)
    A0 = np.nanmedian(y_fit)
    w0 = max(1.0, (x_fit[-1] - x_fit[0]) / 6.0)
    R0, a0, x00 = 0.0, 0.1, np.mean(x_fit)
    p0 = [A0, w0, R0, a0, x00]
    bounds = ([0, 1e-6, -10, -5, x_fit[0]],
              [np.inf, (x_fit[-1]-x_fit[0])*5, 10, 5, x_fit[-1]])

    def resid(p, xx, yy):
        return inv_model(xx, *p) - yy

    res = least_squares(resid, p0, args=(x_fit, y_fit),
                        bounds=bounds, loss=loss, f_scale=f_scale, max_nfev=8000)

    # --- 6) compute delta_beta from fitted model
    A, w, R, alpha, x0 = res.x
    xL, xR = x0 - compute_at_offset*w, x0 + compute_at_offset*w
    invL, invR = inv_model(xL, *res.x), inv_model(xR, *res.x)
    eps = 1e-12
    betaL, betaR = 1.0 / max(invL, eps), 1.0 / max(invR, eps)
    delta_beta = betaR - betaL

    return dict(
        res=res, popt=res.x,
        x_fit=x_fit, y_fit=y_fit,
        region_clean=(s_clean, e_clean),
        region_expanded=(s_expanded, e_expanded),
        x_clean=x_clean, inv_clean=inv_clean, inv_proc=inv,
        inv_left=invL, inv_right=invR,
        beta_left=betaL, beta_right=betaR,
        delta_beta=delta_beta
    )

#####====================================================================================#######
#####======== This is to fit all the pressure, i.e, gas or plasma pressure, =============#######
#####======== magnetic pressure and total pressure                          =============#######
#####====================================================================================#######
import numpy as np
from scipy.optimize import curve_fit

# ====================================================
# Models
# ====================================================

def model_Pmag(x, P0, x0, w, R):
    return P0 * (np.tanh((x - x0) / w) + R) ** 2

def model_Pgas_fixedx0(x, P0, alpha, w, x0):
    xi = np.tanh((x - x0) / w)
    return P0 * (1 - alpha * xi - alpha * xi**2)

def model_Ptot(x, A, B, C, x0, w):
    t = np.tanh((x - x0) / w)
    return A + B * t + C * t**2


# ====================================================
# Helpers
# ====================================================

def _initial_guess_for_tanh(x, y):
    dy = np.gradient(y, x)
    x0_guess = x[np.nanargmax(np.abs(dy))]
    w_guess = max((x[-1] - x[0]) / 10.0, 1e-6)
    return x0_guess, w_guess

def _compute_asymptotic_difference(x, y, frac=0.05):
    n = len(x)
    n_side = max(2, int(frac * n))
    return abs(np.nanmean(y[:n_side]) - np.nanmean(y[-n_side:]))


def _find_three_point_anchors(x, y, x0):
    """
    Identify the local minimum near x0 and two local maxima on either side.
    Returns (x_min, y_min), (x_left_max, y_left_max), (x_right_max, y_right_max)
    """
    # Compute first derivative
    idx_min = np.argmin(np.abs(x - x0))
    window = int(0.1 * len(x))  # search in ~10% window
    lo = max(0, idx_min - window)
    hi = min(len(x), idx_min + window)
    subx, suby = x[lo:hi], y[lo:hi]

    # local min near x0
    idx_local_min = np.argmin(suby)
    x_min, y_min = subx[idx_local_min], suby[idx_local_min]

    # local maxima on either side
    left_mask = subx < x_min
    right_mask = subx > x_min

    if np.any(left_mask):
        idx_left_max = np.argmax(suby[left_mask])
        x_left_max = subx[left_mask][idx_left_max]
        y_left_max = suby[left_mask][idx_left_max]
    else:
        x_left_max, y_left_max = subx[0], suby[0]

    if np.any(right_mask):
        idx_right_max = np.argmax(suby[right_mask])
        x_right_max = subx[right_mask][idx_right_max]
        y_right_max = suby[right_mask][idx_right_max]
    else:
        x_right_max, y_right_max = subx[-1], suby[-1]

    return (x_min, y_min), (x_left_max, y_left_max), (x_right_max, y_right_max)


# ====================================================
# Main fitting function
# ====================================================

def fit_pressures_all(
    x, P_mag, P_gas,
    Pmag_thresh=0.25, Pgas_thresh=0.1, asymp_frac=0.05,
    x0_prev=None, dx_limit=None,
):
    """
    Fit magnetic, gas, and total pressures.

    - P_mag: tanh^2 form
    - P_gas: uses x0 from P_mag and passes through local max near x0
    - P_tot: two fits
        (1) sum of fits  -> ΔPtot_sum
        (2) direct fit   -> passes through local min near x0
    """

    x = np.asarray(x)
    P_mag = np.asarray(P_mag)
    P_gas = np.asarray(P_gas)
    assert len(x) == len(P_mag) == len(P_gas)

    # ---------- Step 1: Fit P_mag ----------
    if x0_prev is not None:
        x0_guess = x0_prev
    else:
        x0_guess = x[np.nanargmin(P_mag)]

    dy = np.gradient(P_mag, x)
    x0_grad = x[np.nanargmax(np.abs(dy))]
    if abs(x0_grad - x0_guess) < 0.1 * (x[-1] - x[0]):
        x0_guess = x0_grad
    if x0_prev is not None and dx_limit is not None:
        x0_guess = np.clip(x0_guess, x0_prev - dx_limit, x0_prev + dx_limit)

    w_guess = max((x[-1] - x[0]) / 10.0, 1e-6)
    P0_guess = np.nanmax(P_mag)
    R_guess = 0.0
    bounds_mag = ([0, x.min(), 1e-6, -10],
                  [np.inf, x.max(), (x.max() - x.min()) * 2, 10])
    popt_mag, _ = curve_fit(model_Pmag, x, P_mag,
                            p0=[P0_guess, x0_guess, w_guess, R_guess],
                            bounds=bounds_mag, maxfev=20000)
    Pmag_fit = model_Pmag(x, *popt_mag)
    ΔPmag = _compute_asymptotic_difference(x, Pmag_fit, frac=asymp_frac)
    if ΔPmag < Pmag_thresh:
        raise RuntimeError(f"P_mag fit rejected: ΔPmag={ΔPmag:.3f} < {Pmag_thresh}")
    x0_fit, w_fit = popt_mag[1], popt_mag[2]


    # ---------- Step 2: Fit P_gas (fixed x0) ----------
    x0_fixed = x0_fit
    w_mag = w_fit
    region_mask = np.abs(x - x0_fixed) < 4 * abs(w_mag)
    if np.count_nonzero(region_mask) < 5:
        region_mask[:] = True
    x_sub, Pgas_sub = x[region_mask], P_gas[region_mask]

    local_window = np.abs(x_sub - x0_fixed) < 2 * abs(w_mag)
    if np.any(local_window):
        idx_anchor = np.argmax(Pgas_sub[local_window])
        x_anchor = x_sub[local_window][idx_anchor]
        P_anchor = Pgas_sub[local_window][idx_anchor]
    else:
        idx_anchor = np.argmax(Pgas_sub)
        x_anchor = x_sub[idx_anchor]
        P_anchor = Pgas_sub[idx_anchor]

    P0_guess_gas = P_anchor
    alpha_guess = 0.2
    w_guess_gas = w_mag
    bounds_gas = ([0, -5, 1e-6], [np.inf, 5, (x.max() - x.min()) * 2])

    def model_gas_fixed(x, P0, alpha, w):
        return model_Pgas_fixedx0(x, P0, alpha, w, x0_fixed)

    popt_gas, _ = curve_fit(model_gas_fixed, x_sub, Pgas_sub,
                            p0=[P0_guess_gas, alpha_guess, w_guess_gas],
                            bounds=bounds_gas, maxfev=20000)
    # enforce passage through anchor
    P_pred_anchor = model_Pgas_fixedx0(np.array([x_anchor]), *popt_gas, x0_fixed)[0]
    if P_pred_anchor != 0:
        popt_gas[0] *= P_anchor / P_pred_anchor

    Pgas_fit = model_Pgas_fixedx0(x, *popt_gas, x0_fixed)
    ΔPgas = _compute_asymptotic_difference(x, Pgas_fit, frac=asymp_frac)
    if ΔPgas < Pgas_thresh:
        raise RuntimeError(f"P_gas fit rejected: ΔPgas={ΔPgas:.3f} < {Pgas_thresh}")


    # ---------- Step 3: Total pressure fits ----------
    P_tot = P_mag + P_gas

    # (a) Sum of fits
    Ptot_from_sum = Pmag_fit + Pgas_fit
    ΔPtot_sum = _compute_asymptotic_difference(x, Ptot_from_sum, frac=asymp_frac)

    # (b) Direct fit around x0_fixed, pass through local min
    local_window_tot = np.abs(x - x0_fixed) < 2 * abs(w_mag)
    if np.any(local_window_tot):
        idx_min = np.argmin(P_tot[local_window_tot])
        x_min = x[local_window_tot][idx_min]
        P_min = P_tot[local_window_tot][idx_min]
    else:
        idx_min = np.argmin(P_tot)
        x_min = x[idx_min]
        P_min = P_tot[idx_min]

    A_guess = np.nanmean(P_tot)
    B_guess = (np.nanmax(P_tot) - np.nanmin(P_tot)) / 2
    C_guess = B_guess / 2
    bounds_tot = ([-np.inf, -np.inf, -np.inf, 1e-6],
                  [np.inf, np.inf, np.inf, (x.max()-x.min())*2])

    def model_Ptot_fixedx0(x, A, B, C, w):
        return model_Ptot(x, A, B, C, x0_fixed, w)

    popt_tot, _ = curve_fit(model_Ptot_fixedx0, x, P_tot,
                            p0=[A_guess, B_guess, C_guess, w_mag],
                            bounds=bounds_tot, maxfev=20000)

    # enforce passage through local min
    P_pred_min = model_Ptot_fixedx0(np.array([x_min]), *popt_tot)[0]
    if P_pred_min != 0:
        popt_tot[0] += (P_min - P_pred_min)  # shift A so curve crosses the min

    Ptot_fit = model_Ptot_fixedx0(x, *popt_tot)
    ΔPtot_fit = _compute_asymptotic_difference(x, Ptot_fit, frac=asymp_frac)


    # # ---------- Step 3: Total pressure fits ----------
    # P_tot = P_mag + P_gas

    # # (a) Sum of fits
    # Ptot_from_sum = Pmag_fit + Pgas_fit
    # ΔPtot_sum = _compute_asymptotic_difference(x, Ptot_from_sum, frac=asymp_frac)

    # # (b) Direct fit — enforce passing through min + two side maxima
    # (x_min, P_min), (x_left, P_left), (x_right, P_right) = _find_three_point_anchors(x, P_tot, x0_fixed)

    # A_guess = np.nanmean(P_tot)
    # B_guess = (np.nanmax(P_tot) - np.nanmin(P_tot)) / 2
    # C_guess = B_guess / 2
    # bounds_tot = ([-np.inf, -np.inf, -np.inf, 1e-6],
    #               [np.inf, np.inf, np.inf, (x.max()-x.min())*2])

    # def model_Ptot_fixedx0(x, A, B, C, w):
    #     return model_Ptot(x, A, B, C, x0_fixed, w)

    # popt_tot, _ = curve_fit(model_Ptot_fixedx0, x, P_tot,
    #                         p0=[A_guess, B_guess, C_guess, w_mag],
    #                         bounds=bounds_tot, maxfev=20000)

    # # ---- Enforce passage through the 3 anchor points ----
    # t_min = np.tanh((x_min - x0_fixed)/popt_tot[3])
    # t_left = np.tanh((x_left - x0_fixed)/popt_tot[3])
    # t_right = np.tanh((x_right - x0_fixed)/popt_tot[3])

    # T = np.array([
    #     [1, t_left, t_left**2],
    #     [1, t_min, t_min**2],
    #     [1, t_right, t_right**2],
    # ])
    # Y = np.array([P_left, P_min, P_right])
    # A_adj, B_adj, C_adj = np.linalg.solve(T, Y)

    # popt_tot[:3] = [A_adj, B_adj, C_adj]
    # Ptot_fit = model_Ptot_fixedx0(x, *popt_tot)
    # ΔPtot_fit = _compute_asymptotic_difference(x, Ptot_fit, frac=asymp_frac)

    # ---------- Step 4: Return ----------
    return {
        "x": x,
        "P_mag_data": P_mag, "P_gas_data": P_gas, "P_tot_data": P_tot,
        "Pmag_fit": Pmag_fit, "Pgas_fit": Pgas_fit,
        "Ptot_from_sum": Ptot_from_sum, "Ptot_fit": Ptot_fit,
        "Pmag_params": {"P0": popt_mag[0], "x0": x0_fit, "w": w_fit, "R": popt_mag[3]},
        "Pgas_params": {"P0": popt_gas[0], "alpha": popt_gas[1],
                        "w": popt_gas[2], "x0": x0_fixed},
        "Ptot_params": {"A": popt_tot[0], "B": popt_tot[1],
                        "C": popt_tot[2], "x0": x0_fixed, "w": popt_tot[3]},
        "ΔPmag": ΔPmag, "ΔPgas": ΔPgas,
        "ΔPtot_sum": ΔPtot_sum, "ΔPtot_fit": ΔPtot_fit,
        "x0_guess": x0_guess, "x0_fixed": x0_fixed, "w_mag": w_mag,
        "x_anchor": x_anchor, "P_anchor": P_anchor,
        "x_min_tot": x_min, "P_min_tot": P_min,
        # "anchors": {
        #     "x_min": x_min, "P_min": P_min,
        #     "x_left": x_left, "P_left": P_left,
        #     "x_right": x_right, "P_right": P_right,
        # }
    }



#####################################################################################################################
###### Calculation of 2d thermal pressure. Although the platform has already been set earilier, #####################
###### to extract the related information was cumbersome. That is why the following function.  #####################
#####################################################################################################################



def _pb_get_axis(diag, axis_name, ts):
    """
    Robustly get axis coordinates from a ParticleBinning diagnostic.
    Smilei/happi provides ParticleBinning.getAxis(axis, timestep). :contentReference[oaicite:3]{index=3}
    """
    # Try the documented signature first
    try:
        arr = np.array(diag.getAxis(axis_name, timestep=ts))
        if arr.size > 0:
            return arr
    except Exception:
        pass

    # Try without timestep (some happi versions accept it)
    try:
        arr = np.array(diag.getAxis(axis_name))
        if arr.size > 0:
            return arr
    except Exception:
        pass

    return None


def _fetch_pb_2d(S, idx, ts):
    """
    Fetch one ParticleBinning diag (2D) at timestep ts.
    Returns: (data2d, x_axis, y_axis)
    """
    diag = S.ParticleBinning(idx, timesteps=ts)
    raw = np.array(diag.getData())   # list -> array, typically (1, nx, ny)
    data = np.squeeze(raw)

    if data.ndim != 2:
        raise ValueError(f"Diagnostic idx={idx} did not return 2D data. Got shape {data.shape}")

    x_axis = _pb_get_axis(diag, "x", ts)
    y_axis = _pb_get_axis(diag, "y", ts)

    # Fallback to indices if axes not available
    if x_axis is None:
        x_axis = np.arange(data.shape[0])
    if y_axis is None:
        y_axis = np.arange(data.shape[1])

    return data.astype(float), x_axis, y_axis


def _safe_divide(num, den, n_floor=1e-30):
    """
    elementwise num/den with safe handling when den is too small.
    """
    out = np.zeros_like(num, dtype=float)
    mask = np.abs(den) > n_floor
    np.divide(num, den, out=out, where=mask)
    return out


def _species_thermal_pressure_2d(n, jx, jy, jz, pxx, pyy, pzz, q, n_floor=1e-30, subtract_bulk=True):
    """
    Compute species velocities and thermal (bulk-subtracted) pressures in 2D.

    q: species charge (electrons: -1, ions: +Z)
    """
    # Bulk velocity from j = q n v  -> v = j/(q n)
    denom = q * n
    vx = _safe_divide(jx, denom, n_floor=n_floor)
    vy = _safe_divide(jy, denom, n_floor=n_floor)
    vz = _safe_divide(jz, denom, n_floor=n_floor)

    if subtract_bulk:
        Pxx_th = pxx - n * vx * vx
        Pyy_th = pyy - n * vy * vy
        Pzz_th = pzz - n * vz * vz
    else:
        Pxx_th, Pyy_th, Pzz_th = pxx, pyy, pzz

    P_th = (Pxx_th + Pyy_th + Pzz_th) / 3.0

    return {
        "vx": vx, "vy": vy, "vz": vz,
        "Pxx_th": Pxx_th, "Pyy_th": Pyy_th, "Pzz_th": Pzz_th,
        "P_th": P_th,
    }


def compute_2d_thermal_pressures_from_particlebinning(
    S, frame, time_steps,
    *,
    particle_map,
    q_e=-1.0,
    q_i=+1.0,          # set to +Z if your ions have charge Z
    n_floor=1e-30,
    subtract_bulk=True
):
    """
    Main entry point:
    - loads needed ParticleBinning diags in 2D
    - computes Pe_th(x,y), Pi_th(x,y), Pth_tot(x,y)

    Returns dict containing axes + pressure fields.
    """
    ts = time_steps[frame]

    needed = [
        "e_n", "e_jx", "e_jy", "e_jz", "e_pxx", "e_pyy", "e_pzz",
        "i_n", "i_jx", "i_jy", "i_jz", "i_pxx", "i_pyy", "i_pzz",
    ]
    for k in needed:
        if k not in particle_map:
            raise KeyError(f"'{k}' missing in PARTICLE_MAP")

    # --- Load electron fields (use e_n for axes reference) ---
    e_n, x_axis, y_axis = _fetch_pb_2d(S, particle_map["e_n"], ts)
    e_jx, _, _ = _fetch_pb_2d(S, particle_map["e_jx"], ts)
    e_jy, _, _ = _fetch_pb_2d(S, particle_map["e_jy"], ts)
    e_jz, _, _ = _fetch_pb_2d(S, particle_map["e_jz"], ts)
    e_pxx, _, _ = _fetch_pb_2d(S, particle_map["e_pxx"], ts)
    e_pyy, _, _ = _fetch_pb_2d(S, particle_map["e_pyy"], ts)
    e_pzz, _, _ = _fetch_pb_2d(S, particle_map["e_pzz"], ts)

    # --- Load ion fields ---
    i_n, x2, y2 = _fetch_pb_2d(S, particle_map["i_n"], ts)
    i_jx, _, _ = _fetch_pb_2d(S, particle_map["i_jx"], ts)
    i_jy, _, _ = _fetch_pb_2d(S, particle_map["i_jy"], ts)
    i_jz, _, _ = _fetch_pb_2d(S, particle_map["i_jz"], ts)
    i_pxx, _, _ = _fetch_pb_2d(S, particle_map["i_pxx"], ts)
    i_pyy, _, _ = _fetch_pb_2d(S, particle_map["i_pyy"], ts)
    i_pzz, _, _ = _fetch_pb_2d(S, particle_map["i_pzz"], ts)

    # Basic shape check (important!)
    if (e_n.shape != i_n.shape) or (e_n.shape != e_pxx.shape) or (i_n.shape != i_pxx.shape):
        raise ValueError(
            "Mismatch in returned shapes. "
            f"e_n={e_n.shape}, i_n={i_n.shape}, e_pxx={e_pxx.shape}, i_pxx={i_pxx.shape}"
        )
    if (x_axis.shape != x2.shape) or (y_axis.shape != y2.shape):
        # not fatal for math, but good to flag
        print("Warning: electron and ion axes differ; using electron axes as reference.")

    # --- Compute thermal pressures ---
    e_out = _species_thermal_pressure_2d(
        e_n, e_jx, e_jy, e_jz, e_pxx, e_pyy, e_pzz,
        q=q_e, n_floor=n_floor, subtract_bulk=subtract_bulk
    )
    i_out = _species_thermal_pressure_2d(
        i_n, i_jx, i_jy, i_jz, i_pxx, i_pyy, i_pzz,
        q=q_i, n_floor=n_floor, subtract_bulk=subtract_bulk
    )

    Pe_th = e_out["P_th"]
    Pi_th = i_out["P_th"]
    Pth_tot = Pe_th + Pi_th

    return {
        "timestep": ts,
        "x": x_axis,
        "y": y_axis,

        "Pe_th": Pe_th,
        "Pi_th": Pi_th,
        "Pth_tot": Pth_tot,

        # components (often useful)
        "Pe_xx_th": e_out["Pxx_th"],
        "Pe_yy_th": e_out["Pyy_th"],
        "Pe_zz_th": e_out["Pzz_th"],
        "Pi_xx_th": i_out["Pxx_th"],
        "Pi_yy_th": i_out["Pyy_th"],
        "Pi_zz_th": i_out["Pzz_th"],

        # velocities (debug / physics checks)
        "e_vx": e_out["vx"], "e_vy": e_out["vy"], "e_vz": e_out["vz"],
        "i_vx": i_out["vx"], "i_vy": i_out["vy"], "i_vz": i_out["vz"],

        # raw densities too
        "e_n": e_n,
        "i_n": i_n,

        #current density
        "i_jz": i_jz,
        "e_jz": e_jz,
    }
###................................................###
###................................................###

######################################################################################################################
#######       This is to calculate the rms and mean thermal pressure and magnetic pressure inside the CS       #######
######################################################################################################################
def _infer_xy_order(field2d, x_grid, y_grid):
    """
    Returns 'xy' if field2d is indexed as [ix, iy] with shape (Nx, Ny),
    returns 'yx' if indexed as [iy, ix] with shape (Ny, Nx).
    """
    Nx, Ny = len(x_grid), len(y_grid)
    if field2d.shape == (Nx, Ny):
        return "xy"
    if field2d.shape == (Ny, Nx):
        return "yx"
    raise ValueError(
        f"Field shape {field2d.shape} is incompatible with "
        f"len(x)={Nx}, len(y)={Ny}. Expected (Nx,Ny) or (Ny,Nx)."
    )

def _make_current_sheet_mask(x_grid, y_grid, x0, w, k=2.0, y_min=None, y_max=None, order="xy"):
    """
    Creates a boolean mask for the region:
        x in [x0 - k|w|, x0 + k|w|]
        y in [y_min, y_max]  (defaults to full y range)
    """
    x_grid = np.asarray(x_grid)
    y_grid = np.asarray(y_grid)

    halfwidth = float(k) * abs(float(w))
    x_lo, x_hi = x0 - halfwidth, x0 + halfwidth

    if y_min is None:
        y_min = float(np.min(y_grid))
    if y_max is None:
        y_max = float(np.max(y_grid))

    x_mask = (x_grid >= x_lo) & (x_grid <= x_hi)
    y_mask = (y_grid >= y_min) & (y_grid <= y_max)

    if order == "xy":      # shape (Nx, Ny)
        mask2d = x_mask[:, None] & y_mask[None, :]
    elif order == "yx":    # shape (Ny, Nx)
        mask2d = y_mask[:, None] & x_mask[None, :]
    else:
        raise ValueError("order must be 'xy' or 'yx'")

    return mask2d, (x_lo, x_hi, y_min, y_max)

def _masked_stats(field2d, mask2d):
    """
    Returns mean and RMS metrics over the masked region.
    - rms_abs: sqrt(mean(P^2))  (RMS relative to zero)
    - rms_fluct: sqrt(mean((P - mean(P))^2)) (RMS fluctuation = std if ddof=0)
    """
    vals = field2d[mask2d]
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "rms_abs": np.nan,
            "rms_fluct": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    mean = float(np.mean(vals))
    rms_abs = float(np.sqrt(np.mean(vals**2)))
    rms_fluct = float(np.sqrt(np.mean((vals - mean)**2)))

    return {
        "n": int(vals.size),
        "mean": mean,
        "rms_abs": rms_abs,
        "rms_fluct": rms_fluct,
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }

def current_sheet_pressure_stats(
    Pmag_2d, Pgas_2d,
    x_grid, y_grid,
    x0, w,
    k=2.0,
    y_min=None, y_max=None,
):
    """
    Compute stats of Pmag and Pgas inside a current-sheet region defined by (x0, w, k).
    """
    Pmag_2d = np.asarray(Pmag_2d)
    Pgas_2d = np.asarray(Pgas_2d)
    x_grid = np.asarray(x_grid)
    y_grid = np.asarray(y_grid)

    if Pmag_2d.shape != Pgas_2d.shape:
        raise ValueError(f"Pmag shape {Pmag_2d.shape} != Pgas shape {Pgas_2d.shape}")

    order = _infer_xy_order(Pmag_2d, x_grid, y_grid)
    mask2d, bounds = _make_current_sheet_mask(
        x_grid, y_grid, x0, w, k=k, y_min=y_min, y_max=y_max, order=order
    )

    # Optionally enforce both finite simultaneously:
    mask2d = mask2d & np.isfinite(Pmag_2d) & np.isfinite(Pgas_2d)

    stats_mag = _masked_stats(Pmag_2d, mask2d)
    stats_gas = _masked_stats(Pgas_2d, mask2d)

    x_lo, x_hi, y_lo, y_hi = bounds
    return {
        "x0": float(x0),
        "w": float(w),
        "k": float(k),
        "order": order,
        "bounds": {"x_lo": float(x_lo), "x_hi": float(x_hi), "y_lo": float(y_lo), "y_hi": float(y_hi)},
        "Pmag": stats_mag,
        "Pgas": stats_gas,
    }
    
################################################################################################################
#######                         To obtain the index for Pmin and Pmax                                   ########
################################################################################################################
import numpy as np
import os

def get_equilibrium_times_from_energies(
    cs_width_di,
    r_value,
    cs_position,
    guide_tag,
    data_dir="./",
    offset=3,
):
    """
    Load energies_fit_values_*.dat and compute:
      - t_Pmin (equilibrium time proxy) using your improved logic
      - t_Pmax (time of peak delta_P_sum, ignoring first `offset` points)

    Parameters
    ----------
    cs_width_di : int or float
        Initial CS width in di units (e.g., 1, 2, 4, 8)
    r_value : float
        r parameter (e.g., 0.125, 0.25, 0.5)
    cs_position : float
        cs position value used in filename (e.g., 0.25)
    guide_tag : str
        "with_guidefield" or "without_guidefield"
    data_dir : str
        Directory containing the energies file
    offset : int
        Ignore first `offset` points when finding max (to avoid early comparable spike)

    Returns
    -------
    P_min_idx : int
    t_Pmin : float
    P_max_idx : int
    t_Pmax : float

    Raises
    ------
    FileNotFoundError if the energies file doesn't exist
    ValueError if the data format is not as expected
    """

    # --- filename tags ---
    r_str = f"{float(r_value):.3f}".replace(".", "p")
    cs_str = f"{float(cs_position):.3f}".replace(".", "p")
    cs_width_str = str(int(cs_width_di)) if float(cs_width_di).is_integer() else str(cs_width_di)

    file_energies = os.path.join(
        data_dir,
        #f"energies_fit_values_r_{r_str}_at_cs_{cs_str}_{guide_tag}_cs_width_{cs_width_str}di_less_time.dat"
        # f"energies_fit_values_length_norm_cs_width_{cs_width_str}di_r_{r_str}_at_cs_{cs_str}_{guide_tag}.dat"
        f"energies_fit_values_length_norm_cs_width_{cs_width_str}di_r_{r_str}_at_cs_{cs_str}_{guide_tag}_reflective_bc.dat"
    )

    if not os.path.exists(file_energies):
        raise FileNotFoundError(f"Energies file not found: {file_energies}")

    data_energies = np.loadtxt(file_energies, comments="#")
    if data_energies.ndim == 1:
        data_energies = data_energies[None, :]  # single row -> 2D

    # Expect at least 15 columns based on your unpacking
    if data_energies.shape[1] < 15:
        raise ValueError(
            f"Unexpected energies file format: {file_energies}\n"
            f"Expected >= 15 columns, got {data_energies.shape[1]}."
        )

    # Unpack like your code
    (
        frame_en, time_en, x0_ini, x0_frame_en, w_mag,
        P_mag_mean, P_gas_mean, Pmag_rms_CS, Pgas_rms_CS,
        delta_P_gas, delta_P_mag, delta_P_sum, delta_P_direct,
        x0_by_x0_ini, w_mag_by_w_mag_ini
    ) = data_energies.T[:15]  # take first 15 columns safely

    time_en = np.asarray(time_en, dtype=float)
    delta_P_sum = np.asarray(delta_P_sum, dtype=float)

    if time_en.size == 0:
        raise ValueError(f"No data rows in energies file: {file_energies}")

    # ---- your logic ----
    P_min_idx_check = int(np.nanargmin(delta_P_sum))
    t_Pmin_candidate = float(time_en[P_min_idx_check])

    offset = int(offset)
    if offset < 0:
        offset = 0
    if offset >= delta_P_sum.size:
        offset = 0  # fallback if too large

    P_max_idx = int(np.nanargmax(delta_P_sum[offset:]) + offset)
    t_Pmax = float(time_en[P_max_idx])

    if t_Pmin_candidate < t_Pmax:
        t_Pmin = t_Pmin_candidate
        P_min_idx = P_min_idx_check
    else:
        subset_before_max = delta_P_sum[:P_max_idx]
        if subset_before_max.size > 0:
            new_min_idx = int(np.nanargmin(subset_before_max))
            t_Pmin = float(time_en[new_min_idx])
            P_min_idx = new_min_idx
        else:
            # max is at the very first element
            t_Pmin = t_Pmax
            P_min_idx = P_max_idx

    return P_min_idx, t_Pmin, P_max_idx, t_Pmax
