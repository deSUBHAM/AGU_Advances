import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import simpson

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
def particle_1d_yavg_safe(S, frame, time_steps, var_list=None):
    """
    Robust 1D y-averaged ParticleBinning loader for Smilei.
    - Skips diagnostics that do not exist.
    - Warns the user for missing diagnostics or missing data.
    - Always returns a dict with keys = requested variables; values = dict with keys "data", "x", "diag".
      If a variable is missing, "data" will be None.
    
    Inputs:
      S   	: happi S object
      frame 	: index into time_steps
      time_steps: list/array of diagnostic timesteps
      var_list  : list of variable names (keys from PARTICLE_MAP)
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

################################################
####### 2d pressure for beta calculation #######   
################################################
import numpy as np

def compute_total_pressure_2d(S, frame, time_step):
    """
    Compute total pressure (electrons + ions) using kinetic theory:
        P_ij = <v_i * v_j> - V_i * V_j
        P_total = (1/3) * trace(P_ij_e + P_ij_i)

    Parameters:
        S          : Simulation object (e.g. from Smilei)
        frame      : Frame index (int)
        time_step  : List or array of time steps corresponding to frames

    Returns:
        total_pressure_2d : 2D numpy array of total pressure
        e_pressure_2d     : 2D electron pressure
        i_pressure_2d     : 2D ion pressure
    """

    # ----------------- Variable names -----------------
    pressure_vars = ["e_pxx", "e_pyy", "e_pzz", "i_pxx", "i_pyy", "i_pzz"]
    current_vars  = ["e_jx", "e_jy", "e_jz", "i_jx", "i_jy", "i_jz"]
    density_vars  = ["e_n", "i_n"]

    # ----------------- Load fields -----------------
    pressure_2d = particle_2d_fields(S, frame, time_step, var_list=pressure_vars)
    current_2d  = particle_2d_fields(S, frame, time_step, var_list=current_vars)
    density_2d  = particle_2d_fields(S, frame, time_step, var_list=density_vars)

    # ----------------- Densities -----------------
    e_n_2d = density_2d["e_n"]["data"]
    i_n_2d = density_2d["i_n"]["data"]

    # ----------------- Currents -----------------
    e_jx_2d = current_2d["e_jx"]["data"]
    e_jy_2d = current_2d["e_jy"]["data"]
    e_jz_2d = current_2d["e_jz"]["data"]

    i_jx_2d = current_2d["i_jx"]["data"]
    i_jy_2d = current_2d["i_jy"]["data"]
    i_jz_2d = current_2d["i_jz"]["data"]

    # ----------------- Drift velocities -----------------
    e_vx_2d = np.zeros_like(e_jx_2d)
    e_vy_2d = np.zeros_like(e_jy_2d)
    e_vz_2d = np.zeros_like(e_jz_2d)
    mask_e = e_n_2d != 0
    e_vx_2d[mask_e] = -e_jx_2d[mask_e] / e_n_2d[mask_e]
    e_vy_2d[mask_e] = -e_jy_2d[mask_e] / e_n_2d[mask_e]
    e_vz_2d[mask_e] = -e_jz_2d[mask_e] / e_n_2d[mask_e]

    i_vx_2d = np.zeros_like(i_jx_2d)
    i_vy_2d = np.zeros_like(i_jy_2d)
    i_vz_2d = np.zeros_like(i_jz_2d)
    mask_i = i_n_2d != 0
    i_vx_2d[mask_i] = i_jx_2d[mask_i] / i_n_2d[mask_i]
    i_vy_2d[mask_i] = i_jy_2d[mask_i] / i_n_2d[mask_i]
    i_vz_2d[mask_i] = i_jz_2d[mask_i] / i_n_2d[mask_i]

    # ----------------- Raw pressures <v_i * v_j> -----------------
    e_pxx_raw_2d = pressure_2d["e_pxx"]["data"]
    e_pyy_raw_2d = pressure_2d["e_pyy"]["data"]
    e_pzz_raw_2d = pressure_2d["e_pzz"]["data"]

    i_pxx_raw_2d = pressure_2d["i_pxx"]["data"]
    i_pyy_raw_2d = pressure_2d["i_pyy"]["data"]
    i_pzz_raw_2d = pressure_2d["i_pzz"]["data"]

    # ----------------- Pressure tensors: P_ij = <v_i*v_j> - V_i*V_j -----------------
    e_pxx_2d = e_pxx_raw_2d - e_vx_2d**2
    e_pyy_2d = e_pyy_raw_2d - e_vy_2d**2
    e_pzz_2d = e_pzz_raw_2d - e_vz_2d**2

    i_pxx_2d = i_pxx_raw_2d - i_vx_2d**2
    i_pyy_2d = i_pyy_raw_2d - i_vy_2d**2
    i_pzz_2d = i_pzz_raw_2d - i_vz_2d**2

    # ----------------- Scalar pressures (1/3 of trace) -----------------
    e_pressure_2d = (1.0 / 3.0) * (e_pxx_2d + e_pyy_2d + e_pzz_2d)
    i_pressure_2d = (1.0 / 3.0) * (i_pxx_2d + i_pyy_2d + i_pzz_2d)

    total_pressure_2d = e_pressure_2d + i_pressure_2d

    return total_pressure_2d, e_pressure_2d, i_pressure_2d

###### curve fitting ######

def sech2(x, A0, x0, w, y0):
    """Numerically stable sech² fit function."""
    z = (x - x0) / w
    return A0 * (1 - np.tanh(z)**2) + y0

def fit_sech2_to_beta(x_data, y_data, window=100, verbose=True):
    """
    Fits beta(x) to a sech² profile near the peak.

    Parameters:
        x_data  : 1D numpy array (x-axis values)
        y_data  : 1D numpy array (beta values)
        window  : Number of points on either side of the peak to include in fit
        verbose : If True, print fit parameters

    Returns:
        fit_params  : [A0, x0, w, y0]
        x_crop      : x values used in the fit
        y_crop      : y values used in the fit
        y_fit_crop  : fitted y values over x_crop
    """

    # Step 1: Find peak index
    peak_idx = np.argmax(y_data)
    peak_x = x_data[peak_idx]

    # Step 2: Define fit window
    start = max(0, peak_idx - window)
    end = min(len(x_data), peak_idx + window)

    x_crop = x_data[start:end]
    y_crop = y_data[start:end]

    # Step 3: Initial guess
    A0_guess = np.max(y_crop)
    x0_guess = peak_x
    w_guess  = 1.0
    y0_guess = np.min(y_crop)
    p0 = [A0_guess, x0_guess, w_guess, y0_guess]

    # Step 4: Curve fitting
    try:
        popt, pcov = curve_fit(sech2, x_crop, y_crop, p0=p0)
    except RuntimeError:
        print("Fit failed. Try adjusting the window or initial guess.")
        return None, x_crop, y_crop, None

    A0, x0, w, y0 = popt
    y_fit_crop = sech2(x_crop, *popt)

    # if verbose:
    #     print(f"Fit Results:")
    #     print(f"  A0 = {A0:.4f}")
    #     print(f"  x0 = {x0:.4f}")
    #     print(f"   w = {w:.4f}")
    #     print(f"  y0 = {y0:.4f}")

    return popt, x_crop, y_crop, y_fit_crop
 



def fit_beta_profile_width(S, frame, time_step, window=100):
    """
    Compute the plasma beta profile and fit the y-averaged beta using a sech² profile.

    Parameters:
        S         : Smilei simulation object
        frame     : Frame index (int)
        time_step : Array of time steps (e.g. from getTimesteps())
        window    : Number of points around the peak to use in fit (default: 100)

    Returns:
        fit_params : List of [A0, x0, w, y0] if fit is successful, else None
        x_crop     : Cropped x-range used for fitting
        y_crop     : Cropped beta profile used for fitting
        y_fit_crop : Fitted beta profile values (sech² fit)
        beta_2d    : Full 2D beta array (useful for plotting)
    """
    # --- Load magnetic field components ---
    Bx = np.squeeze(np.array(S.Field(0, 'Bx').getData(timestep=time_step[frame])), axis=0)
    By = np.squeeze(np.array(S.Field(0, 'By').getData(timestep=time_step[frame])), axis=0)
    Bz = np.squeeze(np.array(S.Field(0, 'Bz').getData(timestep=time_step[frame])), axis=0)

    Bx1 = S.Field(0, 'Bx')
    Bz1 = S.Field(0, 'Bz')

    # --- Get spatial axes and shift to center domain ---
    x_B = Bz1.getAxis('x')
    y_B = Bz1.getAxis('y')
    x_range = x_B - 0.5 * (x_B[0] + x_B[-1])
    y_range = y_B - 0.5 * (y_B[0] + y_B[-1])

    # --- Compute B² ---
    B_sq = Bx**2 + By**2 + Bz**2

    # --- Compute pressure ---
    total_P, e_P, i_P = compute_total_pressure_2d(S, frame, time_step)

    # --- Compute plasma beta ---
    beta_2d = total_P / (B_sq / 2.0)

    # --- Y-average using Simpson's rule ---
    beta_yavg = simpson(beta_2d, x=y_B, axis=1) / (y_B[-1] - y_B[0])

    # --- Fit beta profile using sech² ---
    fit_params, x_crop, y_crop, y_fit_crop = fit_sech2_to_beta(x_range, beta_yavg, window=window)

    return fit_params, x_crop, y_crop, y_fit_crop, beta_2d



def tanh_model(x, B_amp, x0, w, R):
    return B_amp * (np.tanh((x - x0) / w) + R)

def fit_By_tanh(S, time_step, frame, average_axis='y', field_name='By', window=None):
    """
    Fit y-averaged By to B_amp * (tanh((x - x0)/w) + R)
    
    Parameters:
        S           : Smilei simulation object
        time_step   : time_step array from getTimesteps()
        frame       : frame index (int)
        average_axis: axis to average ('y' or 'x')
        field_name  : field to fit (default: 'By')
        window      : (optional) restrict x-range around peak

    Returns:
        fit_params  : [B_amp, x0, w, R] if fit succeeds, else None
        x_fit       : x values used for fit
        y_fit       : y-averaged field values used for fit
        y_fit_model : model values from fit
    """
    # Load averaged field
    field_obj = S.Field(0, field_name, average={average_axis: "all"})
    field_data = np.squeeze(np.array(field_obj.getData(timestep=time_step[frame])), axis=0)
    
    # Get x axis and center it
    x = field_obj.getAxis('x')
    x_centered = x - 0.5 * (x[0] + x[-1])

    y = field_data

    # Optionally crop window around tanh feature
    if window is not None:
        peak_idx = np.argmax(np.abs(y))  # pick where By is strongest
        start = max(0, peak_idx - window)
        end = min(len(x_centered), peak_idx + window)
        x_crop = x_centered[start:end]
        y_crop = y[start:end]
    else:
        x_crop = x_centered
        y_crop = y

    # Initial parameter guess
    B_amp0 = (np.max(y_crop) - np.min(y_crop)) / 2
    x0_0 = x_crop[np.argmax(np.abs(y_crop))]
    w0 = 1.0
    R0 = 0.0

    p0 = [B_amp0, x0_0, w0, R0]

    try:
        popt, _ = curve_fit(tanh_model, x_crop, y_crop, p0=p0, maxfev=10000)
        B_amp, x0, w, R = popt
        y_fit_model = tanh_model(x_crop, *popt)
        return popt, x_crop, y_crop, y_fit_model
    except RuntimeError:
        return None, x_crop, y_crop, None


#####################################################################################
###########                        tanh fit                               ###########
#####################################################################################

import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

def tanh_model(x, B_amp, w, R, x0):
  """Model: tanh(x/w) + R"""
  return B_amp*(np.tanh((x-x0) / w) + R)

def fit_By_tanh_normalized(
  S,
  frame,
  time_step,
  B0,
  di,
  min_variation=1.5,
  window_size=600,
  edge_fraction=0.1,
  smooth_sigma=0.0,
  maxfev=10000
):
  """
  Find and fit a region of y-averaged By/B0 to tanh(x/w) + R.

  Selection criterion: abs(left_mean - right_mean) >= min_variation,
  where left_mean/right_mean are means of the first/last `edge_fraction`
  portion of the window (this avoids fitting small internal wiggles).

  Parameters
  ----------
  S : Smilei simulation object
  frame : int
  time_step : list/array
  B0 : float
  di : float     # ion skin depth, used for initial w guess = 4*di
  min_variation : float
    Required difference between upstream and downstream means (By/B0).
  window_size : int
    Number of x points in the sliding window to consider for fitting.
  edge_fraction : float
    Fraction of window used for computing left/right means (0 < f < 0.5).
  smooth_sigma : float
    If > 0, apply gaussian smoothing with this sigma (in grid points)
    to By before searching (optional; default 0 = no smoothing).
  maxfev : int
    Max function evaluations for curve_fit.

  Returns
  -------
  (x_all, By_norm, x_fit_centered, y_fit, w, R, window_slice)
   x_all : full centered x array
   By_norm : By / B0 (full)
   x_fit_centered : x values used for fit (centered so that fit is symmetric)
   y_fit : data used for fit
   w, R : fit parameters (None if fit failed)
   window_slice : slice(start, end) selected for fit (None if none found)
  """

  # --- Load y-averaged By field
  By1_yav = S.Field(0, 'By', average={"y": "all"})
  time_snap = By1_yav.getTimes()
  time_step = By1_yav.getTimesteps()
  By_data_yav = np.squeeze(np.array(By1_yav.getData(timestep=time_step[frame])), axis=0)
  # time_snap = By1_yav.getTimes()  # if you need time info outside

  # --- Normalize
  By_norm = By_data_yav / B0

  # --- Optional smoothing to reduce small-scale wiggles (off by default)
  if smooth_sigma is not None and smooth_sigma > 0.0:
    By_search = gaussian_filter1d(By_norm, sigma=smooth_sigma)
  else:
    By_search = By_norm.copy()

  # --- Center x-axis
  x = By1_yav.getAxis('x')
  x_centered = x  ##- 0.5 * (x[0] + x[-1])

  N = len(x_centered)
  if window_size >= N:
    raise ValueError("window_size must be smaller than number of x points")

  # --- edge width in points used to compute upstream/downstream means
  edge_width = int(max(3, np.round(edge_fraction * window_size)))
  if edge_width * 2 >= window_size:
    edge_width = max(1, window_size // 10)

  best_start = None
  best_metric = 0.0

  # Slide window and check upstream/downstream mean difference
  for start in range(0, N - window_size + 1):
    end = start + window_size
    y_win = By_search[start:end]

    left_mean = np.mean(y_win[:edge_width])
    right_mean = np.mean(y_win[-edge_width:])
    mean_diff = abs(left_mean - right_mean)

    # Keep the window with the largest mean difference that is >= threshold
    if mean_diff >= min_variation and mean_diff > best_metric:
      best_metric = mean_diff
      best_start = start

  if best_start is None:
    # No region found
    # Return full arrays but None for fit results
    return x_centered, By_norm, None, None, None, None, None, None, None

  # Crop original (unsmoothed) data for the final fit
  start = best_start
  end = start + window_size
  x_fit = x_centered[start:end]
  y_fit = By_norm[start:end]

  # # Center the fit x around zero because model is tanh(x/w)+R with center at 0.
  # x_fit_centered = x_fit - np.mean(x_fit)

  # Initial guesses: w = 4*di, R = average offset between asymptotes
  # Estimate R0 as average of left and right means from original (unsmoothed) window:
  left_mean_real = np.mean(y_fit[:edge_width])
  right_mean_real = np.mean(y_fit[-edge_width:])
  # R0 = 0.5 * (left_mean_real + right_mean_real)
  B_amp = B0
  R0 = 0.5
  w0 = 4.0 * di
  x0_0 = np.mean(x_fit)
  p0 = [B_amp, w0, R0, x0_0]

  try:
    popt, pcov = curve_fit(tanh_model, x_fit, y_fit, p0=p0, maxfev=maxfev)
    B_amp, w, R, x0 = popt
    return x_centered, By_norm, x_fit, y_fit, B_amp, w, R, x0, slice(start, end)
  except RuntimeError:
    # Fit failed; return None for fit params
    return x_centered, By_norm, x_fit, y_fit, None, None, None, None, slice(start, end)
