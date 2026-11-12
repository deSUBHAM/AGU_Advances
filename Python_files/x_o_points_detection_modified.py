import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.markers as MarkerStyle
from numpy.polynomial.legendre import leggauss
from scipy.ndimage import gaussian_filter, label, center_of_mass
from scipy import ndimage

###################################################
#### calculate in a general way: Flux function ####
###################################################
def _orient_to_NzNx(arr, z, x, name="array"):
    """Ensure arr has shape (Nz, Nx). If it's (Nx, Nz), transpose."""
    Nz, Nx = len(z), len(x)
    if arr.shape == (Nz, Nx):
        return arr
    if arr.shape == (Nx, Nz):
        return arr.T
    raise ValueError(
        f"{name} shape {arr.shape} is incompatible with x(len={Nx}), z(len={Nz}). "
        "Expected (Nz, Nx) or (Nx, Nz)."
    )

def _orient_to_NzNx(arr, z, x, name="array"):
    arr = np.asarray(arr)
    Nz, Nx = len(z), len(x)

    # Drop leading singleton axes (e.g., time dimension of size 1)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 2:
        if arr.shape == (Nz, Nx):
            return arr
        if arr.shape == (Nx, Nz):
            return arr.T

    # As a fallback, use the last two axes as spatial
    if arr.ndim >= 2:
        tail = arr.shape[-2:]
        if tail == (Nz, Nx):
            return arr.reshape(-1, Nz, Nx)[-1]
        if tail == (Nx, Nz):
            return arr.reshape(-1, Nx, Nz)[-1].transpose(1, 0)

    raise ValueError(
        f"{name} shape {arr.shape} is incompatible with x(len={Nx}), z(len={Nz}). "
        "Expected (Nz, Nx) or (Nx, Nz), optionally with leading singleton axes."
    )

def general_compute_Ay(x, z, Bx, Bz):
    """
    Compute Ay(x,z) such that:
        ∂Ay/∂z = -Bx  and  ∂Ay/∂x =  Bz
    using trapezoidal cumulative integrals on (possibly) nonuniform grids.
    x: (Nx,), z: (Nz,)
    Bx, Bz: either (Nz, Nx) or (Nx, Nz) — auto-detected.
    Returns Ay with shape (Nz, Nx).
    """
    x = np.asarray(x)
    z = np.asarray(z)
    Nx, Nz = len(x), len(z)
    if Nx < 2 or Nz < 2:
        raise ValueError("x and z must each have length >= 2 for integration.")

    # Ensure fields are (Nz, Nx)
    Bx = _orient_to_NzNx(np.asarray(Bx), z, x, name="Bx")
    Bz = _orient_to_NzNx(np.asarray(Bz), z, x, name="Bz")

    # Grid spacings (nonuniform allowed)
    dx = np.diff(x)            # (Nx-1,)
    dz = np.diff(z)            # (Nz-1,)

    # Output
    Ay = np.zeros((Nz, Nx), dtype=np.result_type(Bx, Bz, float))

    # 1) Integrate -Bx along z at fixed x:
    #    Ay(z_i, x_j) = -∫_{z0}^{z_i} Bx(x_j, z') dz'
    # trapezoid rule column-wise (over z): average Bx between layers times dz
    # integrand_z has shape (Nz-1, Nx)
    integrand_z = 0.5 * (Bx[1:, :] + Bx[:-1, :]) * dz[:, None]
    # cumulative sum along z → shape (Nz-1, Nx); prepend row of zeros at z0
    Ay[1:, :] = -np.cumsum(integrand_z, axis=0)

    # 2) Add integration constant by integrating Bz along x on the bottom row z=z0:
    #    Ay(z0, x_i) = ∫_{x0}^{x_i} Bz(x', z0) dx'
    # trapezoid rule along x on row 0
    Ay[0, 1:] = np.cumsum(0.5 * (Bz[0, 1:] + Bz[0, :-1]) * dx, axis=0)

    # 3) Broadcast that bottom-row constant to all z (gauge fixing)
    Ay[1:, :] += Ay[0, :][None, :]

    # Optional: subtract a global constant (gauge) to center
    # Ay -= Ay.min()  # or Ay -= Ay.mean()

    return Ay


##############################################################################
def to_NyNx(arr, x, y, order=None, name="field"):
    """
    Return arr in shape (Ny, Nx) where axis0=y and axis1=x.

    order:
      - "xy" means arr is indexed as arr[ix, iy] with shape (Nx, Ny) -> transpose
      - "yx" means arr is indexed as arr[iy, ix] with shape (Ny, Nx) -> keep
      - None tries to infer, but if Nx==Ny and arr is square, inference is ambiguous -> raise
    """
    arr = np.squeeze(np.asarray(arr))
    Nx, Ny = len(x), len(y)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D after squeeze; got shape {arr.shape}")

    if order is None:
        # Try to infer from shape
        if arr.shape == (Ny, Nx):
            order = "yx"
        elif arr.shape == (Nx, Ny):
            order = "xy"
        else:
            raise ValueError(
                f"{name} shape {arr.shape} incompatible with (Nx,Ny)=({Nx},{Ny})"
            )

        # If square, inference may still be wrong in practice -> force explicit
        if Nx == Ny and arr.shape == (Nx, Nx):
            raise ValueError(
                f"{name}: Nx==Ny and array is square, orientation is ambiguous. "
                f"Pass order='xy' or order='yx' explicitly."
            )

    if order == "xy":
        if arr.shape != (Nx, Ny):
            raise ValueError(f"{name}: expected shape {(Nx,Ny)} for order='xy', got {arr.shape}")
        return arr.T  # -> (Ny,Nx)

    if order == "yx":
        if arr.shape != (Ny, Nx):
            raise ValueError(f"{name}: expected shape {(Ny,Nx)} for order='yx', got {arr.shape}")
        return arr

    raise ValueError("order must be 'xy', 'yx', or None")


def compute_Az_from_BxBy(x, y, Bx_NyNx, By_NyNx):
    """
    Compute Az(y,x) so that:
      dAz/dy =  Bx
      dAz/dx = -By

    x: (Nx,), y: (Ny,)
    Bx_NyNx, By_NyNx: shape (Ny,Nx)
    returns Az_NyNx: shape (Ny,Nx)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    Bx = np.asarray(Bx_NyNx)
    By = np.asarray(By_NyNx)

    Ny, Nx = Bx.shape
    if By.shape != (Ny, Nx):
        raise ValueError("Bx and By must have same shape (Ny,Nx)")

    dx = np.diff(x)  # (Nx-1,)
    dy = np.diff(y)  # (Ny-1,)

    Az = np.zeros((Ny, Nx), dtype=float)

    # Bottom row y=y0 from integrating -By along x
    Az[0, 1:] = np.cumsum(0.5 * (-(By[0, 1:] + By[0, :-1])) * dx)

    # Then integrate Bx upward along y, preserving the bottom-row constant
    integrand_y = 0.5 * (Bx[1:, :] + Bx[:-1, :]) * dy[:, None]
    Az[1:, :] = Az[0, :][None, :] + np.cumsum(integrand_y, axis=0)

    return Az

##############################################################################
######## Detect all the x and o-points from the Hessian matrix of Ay  ########
##############################################################################

def detect_xo_points(S, time_step, frame, sigma=2, grad_tol_factor=0.02):
    """
    Detect X- and O-points in a 2D reconnection snapshot using Ay derived from Bx, By.
    
    Parameters
    ----------
    S : Smilei Simulation object
        Simulation handle (e.g. from happi.Open())
    time_step : list or array
        List of simulation timesteps corresponding to available frames.
    frame : int
        Frame index (e.g., 0, 1, 2, ...), to choose timestep = time_step[frame].
    sigma : float, optional
        Gaussian smoothing width for Ay (default = 2)
    grad_tol_factor : float, optional
        Fraction of mean gradient magnitude used as threshold for critical point detection.
    
    Returns
    -------
    xX, yX : ndarray
        Arrays of X-point coordinates.
    xO, yO : ndarray
        Arrays of O-point coordinates.
    Ay_smooth, x_range, y_range : ndarray
        Smoothed flux function and grid arrays (for plotting or further use).
    """

    # --- Load B-fields ---
    Bx1 = S.Field(0, 'Bx')
    By1 = S.Field(0, 'By')
    Bz1 = S.Field(0, 'Bz')

    Bx_data = np.squeeze(np.array(Bx1.getData(timestep=time_step[frame])), axis=0)
    By_data = np.squeeze(np.array(By1.getData(timestep=time_step[frame])), axis=0)

    # --- Axes ---
    x_Bz1 = Bz1.getAxis('x')
    y_Bz1 = Bz1.getAxis('y')

    xc_offset = 0.0 # -0.5 * (x_Bz1[0] + x_Bz1[-1])
    yc_offset = 0.0 # -0.5 * (y_Bz1[0] + y_Bz1[-1])
    x_range = x_Bz1 + xc_offset
    y_range = y_Bz1 + yc_offset

    dx = x_Bz1[1] - x_Bz1[0]
    dy = y_Bz1[1] - y_Bz1[0]

    # --- Compute vector potential Ay ---
    Ay = general_compute_Ay(x_range, y_range, Bx_data, By_data)
    # Ay = compute_Az_from_BxBy(x_range, y_range, Bx_data, By_data)

    # --- Smooth Ay ---
    Ay_smooth = gaussian_filter(Ay, sigma=sigma)
    # Ay_smooth = Ay
    
    # --- Compute gradients and Hessian ---
    dAydx = np.gradient(Ay_smooth, dx, axis=1)
    dAydy = np.gradient(Ay_smooth, dy, axis=0)
    Axx = np.gradient(dAydx, dx, axis=1)
    Ayy = np.gradient(dAydy, dy, axis=0)
    Axy = np.gradient(dAydx, dy, axis=0)

    # --- Critical points where grad ~ 0 ---
    grad_mag = np.sqrt(dAydx**2 + dAydy**2)
    tol = grad_tol_factor * np.mean(grad_mag)
    mask_crit = grad_mag < tol

    # --- Eigenvalues for classification ---
    trace = Axx + Ayy
    det = Axx * Ayy - Axy**2
    disc = np.sqrt(np.maximum(trace**2 - 4 * det, 0))
    lambda1 = 0.5 * (trace + disc)
    lambda2 = 0.5 * (trace - disc)

    X_mask = mask_crit & (lambda1 * lambda2 < 0)  # saddle (X-point)
    O_mask = mask_crit & (lambda1 * lambda2 > 0)  # extremum (O-point)

    # --- Cluster detections ---
    def cluster_points(mask, x_range, y_range):
        labeled, num = label(mask)
        coords = center_of_mass(mask, labeled, range(1, num + 1))
        xs = np.array([x_range[int(c[1])] for c in coords])
        ys = np.array([y_range[int(c[0])] for c in coords])
        return xs, ys

    xX, yX = cluster_points(X_mask, x_range, y_range)
    xO, yO = cluster_points(O_mask, x_range, y_range)

    # print(f"[Frame {frame}] Found {len(xX)} X-points and {len(xO)} O-points.")

    # ix_X = np.array([np.argmin(np.abs(x_range - x)) for x in xX_filtered])
    # iy_X = np.array([np.argmin(np.abs(y_range - y)) for y in yX_filtered])
    return xX, yX, xO, yO, Ay_smooth, x_range, y_range
    
####################################################################################################
####################################################################################################
####################################################################################################

# def filter_x_points_by_Ay_mod(Ay_smooth, x_range, y_range, xX, yX, xO, yO, tol_Ay=0.5):
#     """
#     Filter X-points based on their magnetic potential Ay value relative to the 
#     global minimum O-point detected from the O-point set.

#     Parameters
#     ----------
#     Ay_smooth : 2D array
#         Smoothed magnetic flux function.
#     x_range, y_range : 1D arrays
#         Grid coordinates.
#     xX, yX : 1D arrays
#         Detected X-point coordinates (unfiltered).
#     xO, yO : 1D arrays
#         Detected O-point coordinates (unfiltered).
#     tol_Ay : float, optional
#         Threshold for |Ay(X) - Ay_global_min_O|. X-points with smaller differences are discarded.

#     Returns
#     -------
#     xX_filtered, yX_filtered : 1D arrays
#         Filtered X-point coordinates that pass the Ay threshold.
#     Ay_X_filtered : 1D array
#         Corresponding Ay values at the filtered X-points.
#     xO_global, yO_global, Ay_global_min_O : floats
#         Coordinates and value of the global O-point (global minimum of Ay among O-points).
#     """

#     # --- Step 1: Compute Ay at all detected O-points ---
#     Ay_at_O = np.array([
#         Ay_smooth[np.argmin(np.abs(y_range - y)), np.argmin(np.abs(x_range - x))]
#         for x, y in zip(xO, yO)
#     ])

#     # --- Step 2: Identify the global minimum among O-points ---
#     if len(Ay_at_O) == 0:
#         raise ValueError("No O-points detected. Cannot determine global minimum.")
#     idx_global_O = np.argmin(Ay_at_O)
#     Ay_global_min_O = Ay_at_O[idx_global_O]
#     xO_global = xO[idx_global_O]
#     yO_global = yO[idx_global_O]
#     # print(f"Global O-point (from O-set) at (x={xO_global:.3f}, y={yO_global:.3f}), Ay={Ay_global_min_O:.3f}")

#     # --- Step 3: Evaluate Ay at all detected X-points ---
#     Ay_at_X = np.array([
#         Ay_smooth[np.argmin(np.abs(y_range - y)), np.argmin(np.abs(x_range - x))]
#         for x, y in zip(xX, yX)
#     ])

#     # --- Step 4: Apply filtering criterion ---
#     mask_good_X = np.abs(Ay_at_X - Ay_global_min_O) >= tol_Ay
#     xX_filtered = xX[mask_good_X]
#     yX_filtered = yX[mask_good_X]
#     Ay_X_filtered = Ay_at_X[mask_good_X]
#     # print(f"Retained {len(xX_filtered)} of {len(xX)} X-points (tol_Ay = {tol_Ay}).")

#     return xX_filtered, yX_filtered, Ay_X_filtered, xO_global, yO_global, Ay_global_min_O

####################################################################################################
####################################################################################################

import numpy as np

def filter_x_points_by_Ay_mod(
    Ay_smooth, x_range, y_range, xX, yX, xO, yO,
    tol_Ay=0.5,
    missing_O="return_empty",   # "return_empty" | "use_domain_min" | "pass_through"
):
    """
    Filter X-points based on Ay relative to the global minimum O-point (min Ay among O-points).
    If no O-points exist, handle according to missing_O.
    """

    # Make sure inputs are numpy arrays (safe even if already arrays)
    xX = np.asarray(xX)
    yX = np.asarray(yX)
    xO = np.asarray(xO)
    yO = np.asarray(yO)

    def Ay_at_xy(x, y):
        ix = np.argmin(np.abs(x_range - x))
        iy = np.argmin(np.abs(y_range - y))
        return Ay_smooth[iy, ix]   # NOTE: [y, x] indexing

    # ---------- If no O-points detected ----------
    if xO.size == 0 or yO.size == 0:
        if missing_O == "return_empty":
            # Treat as "no valid topology yet": return no X-points
            return (
                np.array([]), np.array([]), np.array([]),
                np.nan, np.nan, np.nan
            )

        elif missing_O == "use_domain_min":
            # Use global minimum of Ay_smooth as a surrogate "O"
            iy_min, ix_min = np.unravel_index(np.nanargmin(Ay_smooth), Ay_smooth.shape)
            Ay_global_min_O = float(Ay_smooth[iy_min, ix_min])
            xO_global = float(x_range[ix_min])
            yO_global = float(y_range[iy_min])

        elif missing_O == "pass_through":
            # Do not filter X; return all X as "filtered"
            Ay_at_X = np.array([Ay_at_xy(x, y) for x, y in zip(xX, yX)]) if xX.size else np.array([])
            return xX, yX, Ay_at_X, np.nan, np.nan, np.nan

        else:
            raise ValueError(f"Unknown missing_O='{missing_O}'")

    else:
        # ---------- Normal case: compute global min among O-points ----------
        Ay_at_O = np.array([Ay_at_xy(x, y) for x, y in zip(xO, yO)])
        # (Ay_at_O cannot be empty here because xO.size>0)
        idx_global_O = np.argmin(Ay_at_O)
        Ay_global_min_O = float(Ay_at_O[idx_global_O])
        xO_global = float(xO[idx_global_O])
        yO_global = float(yO[idx_global_O])

    # ---------- Evaluate Ay at X-points ----------
    if xX.size == 0:
        return np.array([]), np.array([]), np.array([]), xO_global, yO_global, Ay_global_min_O

    Ay_at_X = np.array([Ay_at_xy(x, y) for x, y in zip(xX, yX)])

    # ---------- Filter criterion ----------
    mask_good_X = np.abs(Ay_at_X - Ay_global_min_O) >= tol_Ay
    xX_filtered = xX[mask_good_X]
    yX_filtered = yX[mask_good_X]
    Ay_X_filtered = Ay_at_X[mask_good_X]

    return xX_filtered, yX_filtered, Ay_X_filtered, xO_global, yO_global, Ay_global_min_O

####################################################################################################
####################################################################################################
####################################################################################################

# import numpy as np

def filter_o_points_by_Ay(xO_all, yO_all, xX_filtered, yX_filtered, 
                          Ay_smooth, x_range, y_range, 
                          y_min, y_max, periodic=True, tol=1e-6):
    """
    Filter O-points based on adjacent filtered X-points and Ay values.
    Uses Ay_smooth to compute Ay at O-point coordinates.

    Parameters
    ----------
    xO_all, yO_all : array_like
        Coordinates of all O-points from detect_xo_points().
    xX_filtered, yX_filtered : array_like
        Coordinates of X-points after filtering by filter_x_points_by_Ay().
    Ay_smooth : 2D array
        Smoothed Ay field (Ay[y, x]).
    x_range, y_range : 1D arrays
        Grid coordinates in x and y directions.
    y_min, y_max : float
        Domain boundaries in y-direction.
    periodic : bool, optional
        Whether to treat y boundaries as periodic (default True).
    tol : float, optional
        Numerical tolerance for comparing Ay values.

    Returns
    -------
    xO_filtered, yO_filtered : np.ndarray
        Filtered O-point coordinates.
    alternation_ok : bool
        Whether X- and O-points alternate properly along y.
    non_alternating_indices : list
        Indices of locations where alternation fails.
    """

    # --- Compute Ay at all O-points from the field ---
    AyO_all = np.array([
        Ay_smooth[np.argmin(np.abs(y_range - y)), np.argmin(np.abs(x_range - x))]
        for x, y in zip(xO_all, yO_all)
    ])

    # --- Sort X-points by y ---
    sort_idx = np.argsort(yX_filtered)
    xX_sorted = np.array(xX_filtered)[sort_idx]
    yX_sorted = np.array(yX_filtered)[sort_idx]

    xO_filtered = []
    yO_filtered = []

    # --- Step 1: Filter O-points between adjacent X-points ---
    for i in range(len(yX_sorted) - 1):
        y_low, y_high = yX_sorted[i], yX_sorted[i + 1]
        mask = (yO_all > y_low) & (yO_all < y_high)
        if not np.any(mask):
            continue
        idx = np.argmin(AyO_all[mask])
        xO_filtered.append(xO_all[mask][idx])
        yO_filtered.append(yO_all[mask][idx])

    # --- Step 2: Handle periodic edges ---
    if periodic and len(yX_sorted) >= 2:
        y_high_x = yX_sorted[-1]
        y_low_x  = yX_sorted[0]

        mask_high = (yO_all > y_high_x) & (yO_all <= y_max)
        mask_low  = (yO_all >= y_min) & (yO_all < y_low_x)

        higher_o_idx = np.argmin(AyO_all[mask_high]) if np.any(mask_high) else None
        lower_o_idx  = np.argmin(AyO_all[mask_low])  if np.any(mask_low)  else None

        if higher_o_idx is not None and lower_o_idx is not None:
            Ay_high = AyO_all[mask_high][higher_o_idx]
            Ay_low  = AyO_all[mask_low][lower_o_idx]

            if np.abs(Ay_high - Ay_low) > tol:
                chosen_mask = mask_high if Ay_high < Ay_low else mask_low
                chosen_idx  = higher_o_idx if Ay_high < Ay_low else lower_o_idx
            else:
                # Use local gradient magnitude to decide
                dAy_dy, dAy_dx = np.gradient(Ay_smooth)
                def grad_mag(x, y):
                    ix = np.argmin(np.abs(x_range - x))
                    iy = np.argmin(np.abs(y_range - y))
                    return np.hypot(dAy_dx[iy, ix], dAy_dy[iy, ix])

                grad_high = grad_mag(xO_all[mask_high][higher_o_idx], yO_all[mask_high][higher_o_idx])
                grad_low  = grad_mag(xO_all[mask_low][lower_o_idx],  yO_all[mask_low][lower_o_idx])

                chosen_mask = mask_high if grad_high < grad_low else mask_low
                chosen_idx  = higher_o_idx if grad_high < grad_low else lower_o_idx

            xO_filtered.append(xO_all[chosen_mask][chosen_idx])
            yO_filtered.append(yO_all[chosen_mask][chosen_idx])

    # --- Step 3: Check alternation of X and O points ---
    y_combined = np.concatenate([yX_sorted, yO_filtered])
    types = np.concatenate([np.full_like(yX_sorted, 'X', dtype='<U1'),
                            np.full_like(yO_filtered, 'O', dtype='<U1')])
    sort_idx = np.argsort(y_combined)
    y_ordered = y_combined[sort_idx]
    types_ordered = types[sort_idx]

    non_alternating_indices = [i for i in range(1, len(types_ordered))
                               if types_ordered[i] == types_ordered[i-1]]
    alternation_ok = len(non_alternating_indices) == 0

    if not alternation_ok:
        print(f"⚠️ Non-alternating X/O sequence detected at indices: {non_alternating_indices}")

    return (np.array(xO_filtered), np.array(yO_filtered),
            alternation_ok, non_alternating_indices)




##################################################################################################################
####### Here we modified the above code. We were seeing the occurance of a pair of adjecent x-points      ########
####### depending on tol_Ay in the filter_x_points_by_Ay_mod function. We, therefore, put a condition     ########
####### to either discard one of the x-points or to find a o-point in between. We only put the condition  ########
####### for the occurance of a couple of x-points, nor for multiple x-points, neither for any o-points.   ########
##################################################################################################################
def filter_o_points_by_Ay_mod(xO_all, yO_all, xX_filtered, yX_filtered, 
                          Ay_smooth, x_range, y_range, 
                          y_min, y_max, periodic=True, tol=1e-6):
    """
    Filter O-points based on adjacent filtered X-points and Ay values.
    Automatically repairs non-alternating X/O patterns using Ay and distance logic.
    """

    # --- Compute Ay at all O-points and X-points from the field ---
    def Ay_at(x, y):
        ix = np.argmin(np.abs(x_range - x))
        iy = np.argmin(np.abs(y_range - y))
        return Ay_smooth[iy, ix]

    AyO_all = np.array([Ay_at(x, y) for x, y in zip(xO_all, yO_all)])
    AyX_all = np.array([Ay_at(x, y) for x, y in zip(xX_filtered, yX_filtered)])
    Ay_min_O = np.min(AyO_all)

    # --- Sort X-points by y ---
    sort_idx = np.argsort(yX_filtered)
    xX_sorted = np.array(xX_filtered)[sort_idx]
    yX_sorted = np.array(yX_filtered)[sort_idx]
    AyX_sorted = np.array(AyX_all)[sort_idx]

    xO_filtered, yO_filtered = [], []

    # --- Step 1: Filter O-points between adjacent X-points ---
    for i in range(len(yX_sorted) - 1):
        y_low, y_high = yX_sorted[i], yX_sorted[i + 1]
        mask = (yO_all > y_low) & (yO_all < y_high)
        if not np.any(mask):
            continue
        idx = np.argmin(AyO_all[mask])
        xO_filtered.append(xO_all[mask][idx])
        yO_filtered.append(yO_all[mask][idx])

    # --- Step 2: Handle periodic edges ---
    if periodic and len(yX_sorted) >= 2:
        y_high_x = yX_sorted[-1]
        y_low_x  = yX_sorted[0]
        mask_high = (yO_all > y_high_x) & (yO_all <= y_max)
        mask_low  = (yO_all >= y_min) & (yO_all < y_low_x)
        higher_o_idx = np.argmin(AyO_all[mask_high]) if np.any(mask_high) else None
        lower_o_idx  = np.argmin(AyO_all[mask_low])  if np.any(mask_low)  else None

        if higher_o_idx is not None and lower_o_idx is not None:
            Ay_high = AyO_all[mask_high][higher_o_idx]
            Ay_low  = AyO_all[mask_low][lower_o_idx]
            if np.abs(Ay_high - Ay_low) > tol:
                chosen_mask = mask_high if Ay_high < Ay_low else mask_low
                chosen_idx  = higher_o_idx if Ay_high < Ay_low else lower_o_idx
            else:
                dAy_dy, dAy_dx = np.gradient(Ay_smooth)
                def grad_mag(x, y):
                    ix = np.argmin(np.abs(x_range - x))
                    iy = np.argmin(np.abs(y_range - y))
                    return np.hypot(dAy_dx[iy, ix], dAy_dy[iy, ix])
                grad_high = grad_mag(xO_all[mask_high][higher_o_idx], yO_all[mask_high][higher_o_idx])
                grad_low  = grad_mag(xO_all[mask_low][lower_o_idx],  yO_all[mask_low][lower_o_idx])
                chosen_mask = mask_high if grad_high < grad_low else mask_low
                chosen_idx  = higher_o_idx if grad_high < grad_low else lower_o_idx
            xO_filtered.append(xO_all[chosen_mask][chosen_idx])
            yO_filtered.append(yO_all[chosen_mask][chosen_idx])

    # --- Step 3: Alternation check ---
    y_combined = np.concatenate([yX_sorted, yO_filtered])
    types = np.concatenate([np.full_like(yX_sorted, 'X', dtype='<U1'),
                            np.full_like(yO_filtered, 'O', dtype='<U1')])
    sort_idx = np.argsort(y_combined)
    y_ordered = y_combined[sort_idx]
    types_ordered = types[sort_idx]

    non_alternating_indices = [i for i in range(1, len(types_ordered))
                               if types_ordered[i] == types_ordered[i-1]]
    alternation_ok = len(non_alternating_indices) == 0

    # --- Step 4: Repair non-alternation if needed ---
    if not alternation_ok:
        print(f"⚠️ Non-alternating X/O sequence detected at indices: {non_alternating_indices}")
        Lx, Ly = np.ptp(x_range), np.ptp(y_range)
        Nx, Ny = len(x_range), len(y_range)
        dx, dy = Lx / Nx, Ly / Ny
        dmin = np.sqrt(dx**2 + dy**2)

        xX_new, yX_new, AyX_new = list(xX_sorted), list(yX_sorted), list(AyX_sorted)

        for idx in non_alternating_indices:
            if types_ordered[idx] == 'X' and types_ordered[idx-1] == 'X':
                y1, y2 = y_ordered[idx-1], y_ordered[idx]
                x1 = xX_sorted[np.argmin(np.abs(yX_sorted - y1))]
                x2 = xX_sorted[np.argmin(np.abs(yX_sorted - y2))]
                dist = np.hypot(x2 - x1, y2 - y1)

                if dist < dmin:
                    # Too close: remove one X-point
                    Ay1 = Ay_at(x1, y1); Ay2 = Ay_at(x2, y2)
                    diff1 = abs(Ay1 - Ay_min_O)
                    diff2 = abs(Ay2 - Ay_min_O)
                    remove_idx = np.argmin([diff1, diff2])
                    if remove_idx == 0:
                        xX_new.remove(x1); yX_new.remove(y1)
                    else:
                        xX_new.remove(x2); yX_new.remove(y2)
                else:
                    # Far apart: check if any O-point exists between
                    mask_o = (yO_all > y1) & (yO_all < y2)
                    if not np.any(mask_o):
                        # No O-point, remove weaker X
                        Ay1 = Ay_at(x1, y1); Ay2 = Ay_at(x2, y2)
                        diff1 = abs(Ay1 - Ay_min_O)
                        diff2 = abs(Ay2 - Ay_min_O)
                        remove_idx = np.argmin([diff1, diff2])
                        if remove_idx == 0:
                            xX_new.remove(x1); yX_new.remove(y1)
                        else:
                            xX_new.remove(x2); yX_new.remove(y2)
                    else:
                        # Found O-point → include it
                        idx_o = np.argmin(AyO_all[mask_o])
                        xO_filtered.append(xO_all[mask_o][idx_o])
                        yO_filtered.append(yO_all[mask_o][idx_o])

        xX_sorted = np.array(xX_new)
        yX_sorted = np.array(yX_new)

    return (np.array(xO_filtered), np.array(yO_filtered),
            np.array(xX_sorted), np.array(yX_sorted),
            alternation_ok, non_alternating_indices)

############################################################################################
############################################################################################
############################################################################################
# def filter_o_points_by_Ay_mod_more(xO_all, yO_all, xX_filtered, yX_filtered, 
#                           Ay_smooth, x_range, y_range, 
#                           y_min, y_max, periodic=True, tol=1e-6):
#     """
#     Filter O-points based on adjacent filtered X-points and Ay values.
#     Automatically repairs non-alternating X/O patterns using Ay and distance logic,
#     and now accounts for periodic boundaries in the alternation check.
#     """

#     # --- Helper: compute Ay from field ---
#     def Ay_at(x, y):
#         ix = np.argmin(np.abs(x_range - x))
#         iy = np.argmin(np.abs(y_range - y))
#         return Ay_smooth[iy, ix]

#     # --- Precompute ---
#     AyO_all = np.array([Ay_at(x, y) for x, y in zip(xO_all, yO_all)])
#     AyX_all = np.array([Ay_at(x, y) for x, y in zip(xX_filtered, yX_filtered)])
#     Ay_min_O = np.min(AyO_all)

#     # --- Sort X-points by y ---
#     sort_idx = np.argsort(yX_filtered)
#     xX_sorted = np.array(xX_filtered)[sort_idx]
#     yX_sorted = np.array(yX_filtered)[sort_idx]
#     AyX_sorted = np.array(AyX_all)[sort_idx]

#     xO_filtered, yO_filtered = [], []

#     # --- Step 1: O-points between adjacent X-points ---
#     for i in range(len(yX_sorted) - 1):
#         y_low, y_high = yX_sorted[i], yX_sorted[i + 1]
#         mask = (yO_all > y_low) & (yO_all < y_high)
#         if not np.any(mask):
#             continue
#         idx = np.argmin(AyO_all[mask])
#         xO_filtered.append(xO_all[mask][idx])
#         yO_filtered.append(yO_all[mask][idx])

#     # --- Step 2: Handle periodic edges for O selection ---
#     if periodic and len(yX_sorted) >= 2:
#         y_high_x = yX_sorted[-1]
#         y_low_x  = yX_sorted[0]
#         mask_high = (yO_all > y_high_x) & (yO_all <= y_max)
#         mask_low  = (yO_all >= y_min) & (yO_all < y_low_x)

#         higher_o_idx = np.argmin(AyO_all[mask_high]) if np.any(mask_high) else None
#         lower_o_idx  = np.argmin(AyO_all[mask_low])  if np.any(mask_low)  else None

#         if higher_o_idx is not None and lower_o_idx is not None:
#             Ay_high = AyO_all[mask_high][higher_o_idx]
#             Ay_low  = AyO_all[mask_low][lower_o_idx]
#             if np.abs(Ay_high - Ay_low) > tol:
#                 chosen_mask = mask_high if Ay_high < Ay_low else mask_low
#                 chosen_idx  = higher_o_idx if Ay_high < Ay_low else lower_o_idx
#             else:
#                 dAy_dy, dAy_dx = np.gradient(Ay_smooth)
#                 def grad_mag(x, y):
#                     ix = np.argmin(np.abs(x_range - x))
#                     iy = np.argmin(np.abs(y_range - y))
#                     return np.hypot(dAy_dx[iy, ix], dAy_dy[iy, ix])
#                 grad_high = grad_mag(xO_all[mask_high][higher_o_idx], yO_all[mask_high][higher_o_idx])
#                 grad_low  = grad_mag(xO_all[mask_low][lower_o_idx],  yO_all[mask_low][lower_o_idx])
#                 chosen_mask = mask_high if grad_high < grad_low else mask_low
#                 chosen_idx  = higher_o_idx if grad_high < grad_low else lower_o_idx
#             xO_filtered.append(xO_all[chosen_mask][chosen_idx])
#             yO_filtered.append(yO_all[chosen_mask][chosen_idx])

#     # --- Step 3: Alternation check ---
#     y_combined = np.concatenate([yX_sorted, yO_filtered])
#     types = np.concatenate([np.full_like(yX_sorted, 'X', dtype='<U1'),
#                             np.full_like(yO_filtered, 'O', dtype='<U1')])
#     sort_idx = np.argsort(y_combined)
#     y_ordered = y_combined[sort_idx]
#     types_ordered = types[sort_idx]

#     non_alternating_indices = [i for i in range(1, len(types_ordered))
#                                if types_ordered[i] == types_ordered[i-1]]
#     alternation_ok = len(non_alternating_indices) == 0

#     # --- Step 4: Repair non-alternation (non-periodic logic retained) ---
#     if not alternation_ok:
#         print(f"⚠️ Non-alternating X/O sequence detected at indices: {non_alternating_indices}")
#         Lx, Ly = np.ptp(x_range), np.ptp(y_range)
#         Nx, Ny = len(x_range), len(y_range)
#         dx, dy = Lx / Nx, Ly / Ny
#         dmin = np.sqrt(dx**2 + dy**2)

#         xX_new, yX_new, AyX_new = list(xX_sorted), list(yX_sorted), list(AyX_sorted)

#         for idx in non_alternating_indices:
#             if types_ordered[idx] == 'X' and types_ordered[idx-1] == 'X':
#                 y1, y2 = y_ordered[idx-1], y_ordered[idx]
#                 x1 = xX_sorted[np.argmin(np.abs(yX_sorted - y1))]
#                 x2 = xX_sorted[np.argmin(np.abs(yX_sorted - y2))]
#                 dist = np.hypot(x2 - x1, y2 - y1)

#                 if dist < dmin:
#                     Ay1 = Ay_at(x1, y1); Ay2 = Ay_at(x2, y2)
#                     diff1, diff2 = abs(Ay1 - Ay_min_O), abs(Ay2 - Ay_min_O)
#                     if diff1 < diff2:
#                         xX_new.remove(x1); yX_new.remove(y1)
#                     else:
#                         xX_new.remove(x2); yX_new.remove(y2)
#                 else:
#                     mask_o = (yO_all > y1) & (yO_all < y2)
#                     if not np.any(mask_o):
#                         Ay1 = Ay_at(x1, y1); Ay2 = Ay_at(x2, y2)
#                         diff1, diff2 = abs(Ay1 - Ay_min_O), abs(Ay2 - Ay_min_O)
#                         if diff1 < diff2:
#                             xX_new.remove(x1); yX_new.remove(y1)
#                         else:
#                             xX_new.remove(x2); yX_new.remove(y2)
#                     else:
#                         idx_o = np.argmin(AyO_all[mask_o])
#                         xO_filtered.append(xO_all[mask_o][idx_o])
#                         yO_filtered.append(yO_all[mask_o][idx_o])

#         xX_sorted = np.array(xX_new)
#         yX_sorted = np.array(yX_new)

#     # --- Step 5: Periodic alternation correction ---
#     if len(xX_sorted) != 0:
#         if periodic:
#             y_high_x = yX_sorted[-1]
#             y_low_x  = yX_sorted[0]
    
#             mask_high = (yO_filtered > y_high_x) & (yO_filtered <= y_max)
#             mask_low  = (yO_filtered >= y_min) & (yO_filtered < y_low_x)
    
#             n_high, n_low = np.sum(mask_high), np.sum(mask_low)
    
#             if n_high + n_low == 0:
#                 print("⚠️ No filtered O-points across periodic boundaries → alternation incorrect.")
#                 # Find candidate O-points using Ay minima logic
#                 mask_high_all = (yO_all > y_high_x) & (yO_all <= y_max)
#                 mask_low_all  = (yO_all >= y_min) & (yO_all < y_low_x)
#                 if np.any(mask_high_all):
#                     high_idx = np.argmin(AyO_all[mask_high_all])
#                     x_high, y_high = xO_all[mask_high_all][high_idx], yO_all[mask_high_all][high_idx]
#                 else:
#                     x_high, y_high = None, None
#                 if np.any(mask_low_all):
#                     low_idx = np.argmin(AyO_all[mask_low_all])
#                     x_low, y_low = xO_all[mask_low_all][low_idx], yO_all[mask_low_all][low_idx]
#                 else:
#                     x_low, y_low = None, None
    
#                 # Retain the one with minimum Ay
#                 if x_high is not None and x_low is not None:
#                     if Ay_at(x_high, y_high) < Ay_at(x_low, y_low):
#                         xO_filtered.append(x_high); yO_filtered.append(y_high)
#                     else:
#                         xO_filtered.append(x_low); yO_filtered.append(y_low)
#                 elif x_high is not None:
#                     xO_filtered.append(x_high); yO_filtered.append(y_high)
#                 elif x_low is not None:
#                     xO_filtered.append(x_low); yO_filtered.append(y_low)
#             elif n_high + n_low == 1:
#                 # okay, just one O-point across the wrap — fine
#                 pass
#             elif n_high + n_low > 1:
#                 print("⚠️ More than one O-point across boundaries — keeping the one with min(Ay).")
#                 candidates = np.where(mask_high | mask_low)[0]
#                 min_idx = np.argmin([Ay_at(xO_filtered[i], yO_filtered[i]) for i in candidates])
#                 keep_idx = candidates[min_idx]
#                 xO_filtered = [xO_filtered[keep_idx]]
#                 yO_filtered = [yO_filtered[keep_idx]]

#     return (np.array(xO_filtered), np.array(yO_filtered),
#             np.array(xX_sorted), np.array(yX_sorted),
#             alternation_ok, non_alternating_indices)

######---------------------------------######
import numpy as np

def filter_o_points_by_Ay_mod_more(
    xO_all, yO_all, xX_filtered, yX_filtered,
    Ay_smooth, x_range, y_range,
    y_min, y_max, periodic=True, tol=1e-6
):
    """
    Filter O-points based on adjacent filtered X-points and Ay values.
    Automatically repairs non-alternating X/O patterns using Ay and distance logic,
    and accounts for periodic boundaries in the alternation check.

    FIXES vs your original:
      - Removes points by INDEX (never list.remove(value) on floats).
      - Keeps (x,y) pairs aligned.
      - Robust against duplicates and repeated removals.
      - Step-5 periodic masks always operate on numpy arrays (no list comparisons).
    """

    # -------------------- ensure numpy arrays --------------------
    xO_all = np.asarray(xO_all, dtype=float)
    yO_all = np.asarray(yO_all, dtype=float)
    xX_filtered = np.asarray(xX_filtered, dtype=float)
    yX_filtered = np.asarray(yX_filtered, dtype=float)

    x_range = np.asarray(x_range, dtype=float)
    y_range = np.asarray(y_range, dtype=float)

    # Quick exit
    if xX_filtered.size == 0:
        return (np.array([]), np.array([]),
                np.array([]), np.array([]),
                False, [])

    # -------------------- helper: Ay at a point --------------------
    def Ay_at(x, y):
        ix = int(np.argmin(np.abs(x_range - x)))
        iy = int(np.argmin(np.abs(y_range - y)))
        return float(Ay_smooth[iy, ix])

    # -------------------- precompute Ay on candidates --------------------
    if xO_all.size > 0:
        AyO_all = np.array([Ay_at(x, y) for x, y in zip(xO_all, yO_all)], dtype=float)
        Ay_min_O = float(np.min(AyO_all))
    else:
        AyO_all = np.array([], dtype=float)
        Ay_min_O = float(np.nanmin(Ay_smooth))  # fallback if no O-points at all

    AyX_all = np.array([Ay_at(x, y) for x, y in zip(xX_filtered, yX_filtered)], dtype=float)

    # -------------------- sort X points by y --------------------
    sort_idx = np.argsort(yX_filtered)
    xX_sorted = xX_filtered[sort_idx]
    yX_sorted = yX_filtered[sort_idx]
    AyX_sorted = AyX_all[sort_idx]

    # grid-based distance scale (for “too close” checks / dedup)
    dxg = np.nanmedian(np.abs(np.diff(x_range))) if x_range.size > 1 else 1.0
    dyg = np.nanmedian(np.abs(np.diff(y_range))) if y_range.size > 1 else 1.0
    dmin = float(np.hypot(dxg, dyg))
    if not np.isfinite(dmin) or dmin <= 0:
        dmin = 1e-6

    # -------------------- helper: add O if not already present --------------------
    def append_unique_o(xO, yO, x_new, y_new, dist_tol=0.5):
        """Append (x_new,y_new) if not within dist_tol*dmin of an existing O."""
        if xO.size == 0:
            return np.array([x_new]), np.array([y_new])
        dist = np.hypot(xO - x_new, yO - y_new)
        if np.any(dist < dist_tol * dmin):
            return xO, yO
        return np.append(xO, x_new), np.append(yO, y_new)

    # -------------------- Step 1: choose O between adjacent Xs --------------------
    xO_filtered = np.array([], dtype=float)
    yO_filtered = np.array([], dtype=float)

    if yX_sorted.size >= 2 and yO_all.size > 0:
        for i in range(yX_sorted.size - 1):
            y_low, y_high = yX_sorted[i], yX_sorted[i + 1]
            mask = (yO_all > y_low) & (yO_all < y_high)
            if not np.any(mask):
                continue
            idx_local = int(np.argmin(AyO_all[mask]))
            x_pick = float(xO_all[mask][idx_local])
            y_pick = float(yO_all[mask][idx_local])
            xO_filtered, yO_filtered = append_unique_o(xO_filtered, yO_filtered, x_pick, y_pick)

    # -------------------- Step 2: periodic edge O selection (wrap region) --------------------
    if periodic and yX_sorted.size >= 2 and yO_all.size > 0:
        y_high_x = float(yX_sorted[-1])
        y_low_x  = float(yX_sorted[0])

        mask_high = (yO_all > y_high_x) & (yO_all <= y_max)
        mask_low  = (yO_all >= y_min) & (yO_all < y_low_x)

        higher_o_idx = int(np.argmin(AyO_all[mask_high])) if np.any(mask_high) else None
        lower_o_idx  = int(np.argmin(AyO_all[mask_low]))  if np.any(mask_low)  else None

        if higher_o_idx is not None and lower_o_idx is not None:
            Ay_high = float(AyO_all[mask_high][higher_o_idx])
            Ay_low  = float(AyO_all[mask_low][lower_o_idx])

            if np.abs(Ay_high - Ay_low) > tol:
                chosen_mask = mask_high if Ay_high < Ay_low else mask_low
                chosen_idx  = higher_o_idx if Ay_high < Ay_low else lower_o_idx
            else:
                # tie-break by gradient magnitude
                dAy_dy, dAy_dx = np.gradient(Ay_smooth)
                def grad_mag(x, y):
                    ix = int(np.argmin(np.abs(x_range - x)))
                    iy = int(np.argmin(np.abs(y_range - y)))
                    return float(np.hypot(dAy_dx[iy, ix], dAy_dy[iy, ix]))

                xh = float(xO_all[mask_high][higher_o_idx])
                yh = float(yO_all[mask_high][higher_o_idx])
                xl = float(xO_all[mask_low][lower_o_idx])
                yl = float(yO_all[mask_low][lower_o_idx])

                chosen_mask = mask_high if grad_mag(xh, yh) < grad_mag(xl, yl) else mask_low
                chosen_idx  = higher_o_idx if chosen_mask is mask_high else lower_o_idx

            x_pick = float(xO_all[chosen_mask][chosen_idx])
            y_pick = float(yO_all[chosen_mask][chosen_idx])
            xO_filtered, yO_filtered = append_unique_o(xO_filtered, yO_filtered, x_pick, y_pick)

    # -------------------- helper: build ordered sequence with IDs --------------------
    def build_order(yX, yO):
        """
        Returns:
          y_ordered, types_ordered, ids_ordered
        where types_ordered: 0='X', 1='O'
              ids_ordered: index into corresponding X or O arrays.
        """
        nX = yX.size
        nO = yO.size

        if nO == 0:
            y_comb = yX.copy()
            types = np.zeros(nX, dtype=np.int8)
            ids = np.arange(nX, dtype=int)
        else:
            y_comb = np.concatenate([yX, yO])
            types = np.concatenate([np.zeros(nX, dtype=np.int8), np.ones(nO, dtype=np.int8)])
            ids = np.concatenate([np.arange(nX, dtype=int), np.arange(nO, dtype=int)])

        s = np.argsort(y_comb)
        return y_comb[s], types[s], ids[s]

    # -------------------- Step 3/4: alternation repair (iterative) --------------------
    max_iter = 5 * (xX_sorted.size + xO_filtered.size + 1)
    it = 0

    while it < max_iter:
        it += 1

        y_ordered, types_ordered, ids_ordered = build_order(yX_sorted, yO_filtered)
        non_alt = [k for k in range(1, len(types_ordered)) if types_ordered[k] == types_ordered[k-1]]
        if len(non_alt) == 0:
            break  # fixed

        # Fix one issue per iteration (stable)
        k = non_alt[0]
        tcode = int(types_ordered[k])  # 0 X-X, 1 O-O

        if tcode == 0:
            # X-X adjacency: get the two X indices
            id1 = int(ids_ordered[k-1])
            id2 = int(ids_ordered[k])

            x1, y1, Ay1x = float(xX_sorted[id1]), float(yX_sorted[id1]), float(AyX_sorted[id1])
            x2, y2, Ay2x = float(xX_sorted[id2]), float(yX_sorted[id2]), float(AyX_sorted[id2])

            y_low, y_high = (y1, y2) if y1 < y2 else (y2, y1)

            # If there exists an O candidate between them in the full O set, insert it.
            if yO_all.size > 0:
                mask_o = (yO_all > y_low) & (yO_all < y_high)
                if np.any(mask_o):
                    idx_local = int(np.argmin(AyO_all[mask_o]))
                    x_pick = float(xO_all[mask_o][idx_local])
                    y_pick = float(yO_all[mask_o][idx_local])
                    xO_filtered, yO_filtered = append_unique_o(xO_filtered, yO_filtered, x_pick, y_pick)
                    continue  # recompute order

            # Otherwise: remove one X (keep the one "farther" from Ay_min_O, as in your logic)
            diff1 = abs(Ay1x - Ay_min_O)
            diff2 = abs(Ay2x - Ay_min_O)

            remove_id = id1 if diff1 < diff2 else id2

            xX_sorted = np.delete(xX_sorted, remove_id)
            yX_sorted = np.delete(yX_sorted, remove_id)
            AyX_sorted = np.delete(AyX_sorted, remove_id)
            continue

        else:
            # O-O adjacency: keep the one with lower Ay, remove the higher Ay
            id1 = int(ids_ordered[k-1])
            id2 = int(ids_ordered[k])

            x1, y1 = float(xO_filtered[id1]), float(yO_filtered[id1])
            x2, y2 = float(xO_filtered[id2]), float(yO_filtered[id2])

            Ay1o = Ay_at(x1, y1)
            Ay2o = Ay_at(x2, y2)

            remove_id = id1 if Ay1o > Ay2o else id2

            xO_filtered = np.delete(xO_filtered, remove_id)
            yO_filtered = np.delete(yO_filtered, remove_id)
            continue

    # -------------------- Step 5: periodic alternation correction (robust) --------------------
    if periodic and xX_sorted.size >= 2:
        y_high_x = float(yX_sorted[-1])
        y_low_x  = float(yX_sorted[0])

        yO = np.asarray(yO_filtered, dtype=float)

        mask_high = (yO > y_high_x) & (yO <= y_max)
        mask_low  = (yO >= y_min) & (yO < y_low_x)

        n_wrap = int(np.sum(mask_high) + np.sum(mask_low))

        if n_wrap == 0:
            # try to add one wrap O from full candidate set
            if yO_all.size > 0:
                mask_high_all = (yO_all > y_high_x) & (yO_all <= y_max)
                mask_low_all  = (yO_all >= y_min) & (yO_all < y_low_x)

                candidates = []
                if np.any(mask_high_all):
                    idxh = int(np.argmin(AyO_all[mask_high_all]))
                    candidates.append((float(xO_all[mask_high_all][idxh]), float(yO_all[mask_high_all][idxh])))
                if np.any(mask_low_all):
                    idxl = int(np.argmin(AyO_all[mask_low_all]))
                    candidates.append((float(xO_all[mask_low_all][idxl]), float(yO_all[mask_low_all][idxl])))

                if len(candidates) > 0:
                    # keep the one with lower Ay
                    best = min(candidates, key=lambda xy: Ay_at(xy[0], xy[1]))
                    xO_filtered, yO_filtered = append_unique_o(xO_filtered, yO_filtered, best[0], best[1])

        elif n_wrap > 1:
            # keep only the wrap O with minimum Ay
            wrap_idx = np.where(mask_high | mask_low)[0]
            Ay_wrap = np.array([Ay_at(float(xO_filtered[ii]), float(yO_filtered[ii])) for ii in wrap_idx], dtype=float)
            keep = wrap_idx[int(np.argmin(Ay_wrap))]

            keep_mask = np.ones(yO_filtered.size, dtype=bool)
            keep_mask[wrap_idx] = False
            keep_mask[keep] = True

            xO_filtered = xO_filtered[keep_mask]
            yO_filtered = yO_filtered[keep_mask]

    # -------------------- Final alternation check (return values) --------------------
    y_ordered, types_ordered, ids_ordered = build_order(yX_sorted, yO_filtered)
    non_alternating_indices = [k for k in range(1, len(types_ordered)) if types_ordered[k] == types_ordered[k-1]]
    alternation_ok = (len(non_alternating_indices) == 0)

    return (np.asarray(xO_filtered), np.asarray(yO_filtered),
            np.asarray(xX_sorted), np.asarray(yX_sorted),
            alternation_ok, non_alternating_indices)



####################################################################################################################
import numpy as np

def prune_close_o_and_adjacent_x(
    xO, yO,
    xX, yX,
    Ay_smooth, x_range, y_range,
    *,
    dist_tol,
    periodic_y=True,
    y_min=None, y_max=None,
    periodic_x=False,
    x_min=None, x_max=None,
    max_iter=10_000,
    tie_break="closer",   # if Ay equal: remove the closer X to the O
    verbose=False,
):
    """
    Post-filter to remove spurious O/X pairs based on distance to adjacent X-points.

    Logic (as requested):
      - For each O, find the adjacent X points along y (prev/next), with periodic wrap in y.
      - Compute distances from O to each adjacent X.
      - If min(distance) < dist_tol:
            remove that O
            remove one of its adjacent X's (keep the X with higher Ay)
      - Repeat until no violating O remains.
      - Finally ensure each remaining O has two adjacent X's; if not, drop those O's.

    Parameters
    ----------
    xO, yO : 1D arrays
        Filtered O points (output of filter_o_points_by_Ay_mod_more)
    xX, yX : 1D arrays
        Filtered X points (ideally sorted by y, but not required)
    Ay_smooth : 2D array
        Flux function on grid [iy, ix]
    x_range, y_range : 1D arrays
        Grid coordinates
    dist_tol : float
        Distance tolerance in the SAME units as x_range/y_range.
    periodic_y : bool
        Whether y is periodic (adjacency and distance use wrap)
    y_min, y_max : float or None
        Domain bounds in y. If None, inferred from y_range.
    periodic_x : bool
        If x is periodic and you want shortest-wrap distance in x too.
    x_min, x_max : float or None
        Domain bounds in x if periodic_x True. If None, inferred from x_range.
    max_iter : int
        Safety limit on number of prune operations.
    tie_break : {"closer","prev","next"}
        If two adjacent X have equal Ay (within eps), which one to remove.
    verbose : bool
        Print pruning steps.

    Returns
    -------
    xO_new, yO_new, xX_new, yX_new : 1D arrays
        Pruned sets (X sorted by y).
    report : dict
        Diagnostics about what was removed.
    """

    xO = np.asarray(xO, dtype=float)
    yO = np.asarray(yO, dtype=float)
    xX = np.asarray(xX, dtype=float)
    yX = np.asarray(yX, dtype=float)

    if xO.shape != yO.shape or xX.shape != yX.shape:
        raise ValueError("xO/yO and xX/yX must have matching shapes.")
    if dist_tol <= 0:
        raise ValueError("dist_tol must be > 0.")

    if y_min is None: y_min = float(np.min(y_range))
    if y_max is None: y_max = float(np.max(y_range))
    Ly = float(y_max - y_min)

    if periodic_x:
        if x_min is None: x_min = float(np.min(x_range))
        if x_max is None: x_max = float(np.max(x_range))
        Lx = float(x_max - x_min)
    else:
        Lx = None

    # --- Ay at point helper (nearest grid cell) ---
    def Ay_at(x, y):
        ix = int(np.argmin(np.abs(x_range - x)))
        iy = int(np.argmin(np.abs(y_range - y)))
        return float(Ay_smooth[iy, ix])

    # --- periodic delta helper ---
    def dper(delta, L):
        # shortest distance on a periodic interval of length L
        d = np.abs(delta)
        if L <= 0:
            return d
        d = np.mod(d, L)
        return np.minimum(d, L - d)

    def dist(x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        if periodic_y:
            dy = dper(dy, Ly)
        else:
            dy = np.abs(dy)
        if periodic_x:
            dx = dper(dx, Lx)
        else:
            dx = np.abs(dx)
        return float(np.hypot(dx, dy))

    # Nothing to do
    if xO.size == 0 or xX.size == 0:
        return xO.copy(), yO.copy(), xX.copy(), yX.copy(), {
            "removed_O": 0, "removed_X": 0, "reason": "empty_input"
        }

    keepO = np.ones(xO.size, dtype=bool)
    keepX = np.ones(xX.size, dtype=bool)

    removed_O_idx = []
    removed_X_idx = []

    eps = 1e-12
    it = 0

    while it < max_iter:
        it += 1

        # Current kept sets
        O_idx = np.where(keepO)[0]
        X_idx = np.where(keepX)[0]

        if O_idx.size == 0:
            break
        if X_idx.size < 2:
            # cannot give each O two neighbors
            if verbose:
                print("Stopping prune: fewer than 2 X points remain.")
            break

        # Build sorted X by y (for adjacency)
        yXk = yX[X_idx]
        sortX = np.argsort(yXk)
        Xs_idx = X_idx[sortX]                # indices into original X arrays
        yXs = yX[Xs_idx]
        xXs = xX[Xs_idx]
        AyXs = np.array([Ay_at(x, y) for x, y in zip(xXs, yXs)])

        # For each O, find adjacent X prev/next
        # Use y-order insertion point
        viol_best = None  # tuple(min_dist, O_original_idx, prev_pos, next_pos, dprev, dnext)

        for oi in O_idx:
            xo, yo = xO[oi], yO[oi]

            # insertion index in sorted yXs
            ins = int(np.searchsorted(yXs, yo))
            next_pos = ins % yXs.size
            prev_pos = (ins - 1) % yXs.size

            # if somehow prev==next (only possible when size==1, already excluded)
            x_prev, y_prev = xXs[prev_pos], yXs[prev_pos]
            x_next, y_next = xXs[next_pos], yXs[next_pos]

            dprev = dist(xo, yo, x_prev, y_prev)
            dnext = dist(xo, yo, x_next, y_next)

            dmin = dprev if dprev < dnext else dnext

            if dmin < dist_tol:
                if (viol_best is None) or (dmin < viol_best[0]):
                    viol_best = (dmin, oi, prev_pos, next_pos, dprev, dnext)

        # No violations => done
        if viol_best is None:
            break

        dmin, oi, prev_pos, next_pos, dprev, dnext = viol_best

        # Decide which X to remove: keep higher Ay
        Ay_prev = AyXs[prev_pos]
        Ay_next = AyXs[next_pos]

        # Choose to remove the lower-Ay one
        if Ay_prev > Ay_next + eps:
            remove_pos = next_pos
            keep_pos = prev_pos
        elif Ay_next > Ay_prev + eps:
            remove_pos = prev_pos
            keep_pos = next_pos
        else:
            # Tie in Ay: tie-break rule
            if tie_break == "closer":
                # remove the closer X to the O (more likely spurious duplicate)
                remove_pos = prev_pos if dprev < dnext else next_pos
                keep_pos = next_pos if remove_pos == prev_pos else prev_pos
            elif tie_break == "prev":
                remove_pos, keep_pos = prev_pos, next_pos
            elif tie_break == "next":
                remove_pos, keep_pos = next_pos, prev_pos
            else:
                raise ValueError("tie_break must be 'closer', 'prev', or 'next'")

        x_remove_idx = int(Xs_idx[remove_pos])  # original X index to remove

        # Apply removals:
        keepO[oi] = False
        keepX[x_remove_idx] = False

        removed_O_idx.append(int(oi))
        removed_X_idx.append(int(x_remove_idx))

        if verbose:
            print(
                f"[prune] Removed O#{oi} (dmin={dmin:.3g} < {dist_tol}) "
                f"and removed X#{x_remove_idx} (kept neighbor X#{int(Xs_idx[keep_pos])})"
            )

    # Build final kept arrays
    xO_new = xO[keepO]
    yO_new = yO[keepO]
    xX_new = xX[keepX]
    yX_new = yX[keepX]

    # Final check: each O must have two adjacent X points
    # If <2 X remain, drop all O.
    if xX_new.size < 2:
        xO_new = np.array([], dtype=float)
        yO_new = np.array([], dtype=float)

    # Sort X by y before returning (useful downstream)
    if xX_new.size > 0:
        s = np.argsort(yX_new)
        xX_new = xX_new[s]
        yX_new = yX_new[s]

    # Optionally, drop O's that cannot get two distinct neighbors (should only happen if X<2)
    # but we'll do a safe check anyway:
    if xO_new.size > 0 and xX_new.size >= 2:
        yXs = yX_new
        # Remove O if prev==next (should never happen here)
        goodO = np.ones(xO_new.size, dtype=bool)
        for k in range(xO_new.size):
            ins = int(np.searchsorted(yXs, yO_new[k]))
            next_pos = ins % yXs.size
            prev_pos = (ins - 1) % yXs.size
            if prev_pos == next_pos:
                goodO[k] = False
        xO_new = xO_new[goodO]
        yO_new = yO_new[goodO]

    report = {
        "removed_O": len(removed_O_idx),
        "removed_X": len(removed_X_idx),
        "removed_O_indices": np.array(removed_O_idx, dtype=int),
        "removed_X_indices": np.array(removed_X_idx, dtype=int),
        "iters": it,
        "final_nX": int(xX_new.size),
        "final_nO": int(xO_new.size),
    }
    return xO_new, yO_new, xX_new, yX_new, report


###########################################################################################
#####=======               Calculating the E-field at the X-points             =======#####
###########################################################################################

def compute_rms_electric_fields(Ex_data, Ey_data, Ez_data, x_points, y_points, x_range, y_range):
    """
    Compute RMS of both Ez and |E| at specified points.

    Parameters
    ----------
    Ex_data, Ey_data, Ez_data : 2D arrays
        Electric field components (same shape as Ay, Bx, etc.)
    x_points, y_points : 1D arrays
        Coordinates of X-points (filtered & final)
    x_range, y_range : 1D arrays
        Grid coordinates

    Returns
    -------
    Ez_rms : float
        RMS of Ez at X-points (0.0 if no X-points found)
    Emag_rms : float
        RMS of |E| = sqrt(Ex^2 + Ey^2 + Ez^2) at X-points (0.0 if no X-points found)
    Ez_values : 1D array
        Ez at each X-point (empty if no X-points)
    Emag_values : 1D array
        |E| at each X-point (empty if no X-points)
    """

    def nearest_indices(x, y):
        ix = np.argmin(np.abs(x_range - x))
        iy = np.argmin(np.abs(y_range - y))
        return ix, iy

    # --- Case: no X-points detected ---
    if len(x_points) == 0:
        Ez_rms = 0.0
        Emag_rms = 0.0
        Ez_values = np.array([])
        Emag_values = np.array([])
        return Ez_rms, Emag_rms, Ez_values, Emag_values

    # --- Otherwise, compute at all X-points ---
    Ez_values = []
    Emag_values = []

    for x, y in zip(x_points, y_points):
        ix, iy = nearest_indices(x, y)
        Ex_val = Ex_data[ix, iy]
        Ey_val = Ey_data[ix, iy]
        Ez_val = Ez_data[ix, iy]
        Ez_values.append(Ez_val)
        Emag_values.append(np.sqrt(Ex_val**2 + Ey_val**2 + Ez_val**2))

    Ez_values = np.array(Ez_values)
    Emag_values = np.array(Emag_values)
    Ez_rms = np.sqrt(np.mean(Ez_values**2))
    Emag_rms = np.sqrt(np.mean(Emag_values**2))

    return Ez_rms, Emag_rms, Ez_values, Emag_values

####***************************************************####

########################################################################
######  To calculate the reconnected flux or Ay_max - Ay_min      ######
########################################################################
import numpy as np

def Ay_minmax_at_x_nearest(Ay_smooth, x_range, y_range, xline):
    Ay = np.asarray(Ay_smooth)
    x = np.asarray(x_range)
    y = np.asarray(y_range)

    # if someone accidentally gave transposed data, fix it safely
    if Ay.shape == (len(x), len(y)):
        Ay = Ay.T  # -> (Ny, Nx)

    # nearest x index
    ix = int(np.nanargmin(np.abs(x - xline)))

    Ay_slice = Ay[:, ix]  # varies with y at fixed x

    Ay_min = float(np.nanmin(Ay_slice))
    Ay_max = float(np.nanmax(Ay_slice))

    iy_min = int(np.nanargmin(Ay_slice))
    iy_max = int(np.nanargmax(Ay_slice))

    return {
        "xline_requested": float(xline),
        "xline_used": float(x[ix]),
        "ix": ix,
        "Ay_min": Ay_min, "y_at_min": float(y[iy_min]),
        "Ay_max": Ay_max, "y_at_max": float(y[iy_max]),
        "Ay_slice": Ay_slice,   # (Ny,)
    }


