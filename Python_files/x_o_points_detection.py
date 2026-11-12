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
# def _orient_to_NzNx(arr, z, x, name="array"):
#     """Ensure arr has shape (Nz, Nx). If it's (Nx, Nz), transpose."""
#     Nz, Nx = len(z), len(x)
#     if arr.shape == (Nz, Nx):
#         return arr
#     if arr.shape == (Nx, Nz):
#         return arr.T
#     raise ValueError(
#         f"{name} shape {arr.shape} is incompatible with x(len={Nx}), z(len={Nz}). "
#         "Expected (Nz, Nx) or (Nx, Nz)."
#     )

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

    xc_offset = -0.5 * (x_Bz1[0] + x_Bz1[-1])
    yc_offset = -0.5 * (y_Bz1[0] + y_Bz1[-1])
    x_range = x_Bz1 + xc_offset
    y_range = y_Bz1 + yc_offset

    dx = x_Bz1[1] - x_Bz1[0]
    dy = y_Bz1[1] - y_Bz1[0]

    # --- Compute vector potential Ay ---
    Ay = general_compute_Ay(x_range, y_range, Bx_data, By_data)

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


# import numpy as np
# from scipy.ndimage import gaussian_filter, label, center_of_mass

# def detect_xo_points(S, time_step, frame, sigma=2, grad_tol_factor=0.02):
#     """
#     Detect X- and O-points in a 2D reconnection snapshot using Ay derived from Bx, By.
#     Also returns their grid indices for direct field sampling.
    
#     Parameters
#     ----------
#     S : Smilei Simulation object
#         Simulation handle (e.g. from happi.Open())
#     time_step : list or array
#         List of simulation timesteps corresponding to available frames.
#     frame : int
#         Frame index (e.g., 0, 1, 2, ...), to choose timestep = time_step[frame].
#     sigma : float, optional
#         Gaussian smoothing width for Ay (default = 2)
#     grad_tol_factor : float, optional
#         Fraction of mean gradient magnitude used as threshold for critical point detection.
    
#     Returns
#     -------
#     xX, yX, ix_X, iy_X : ndarray
#         X-point coordinates and their integer grid indices.
#     xO, yO, ix_O, iy_O : ndarray
#         O-point coordinates and their integer grid indices.
#     Ay_smooth, x_range, y_range : ndarray
#         Smoothed flux function and grid arrays (for plotting or further use).
#     """

#     # --- Load B-fields ---
#     Bx1 = S.Field(0, 'Bx')
#     By1 = S.Field(0, 'By')
#     Bz1 = S.Field(0, 'Bz')

#     Bx_data = np.squeeze(np.array(Bx1.getData(timestep=time_step[frame])), axis=0)
#     By_data = np.squeeze(np.array(By1.getData(timestep=time_step[frame])), axis=0)

#     # --- Axes ---
#     x_Bz1 = Bz1.getAxis('x')
#     y_Bz1 = Bz1.getAxis('y')

#     xc_offset = -0.5 * (x_Bz1[0] + x_Bz1[-1])
#     yc_offset = -0.5 * (y_Bz1[0] + y_Bz1[-1])
#     x_range = x_Bz1 + xc_offset
#     y_range = y_Bz1 + yc_offset

#     # x_range = x_Bz1
#     # y_range = y_Bz1
    
#     dx = x_Bz1[1] - x_Bz1[0]
#     dy = y_Bz1[1] - y_Bz1[0]

#     # --- Compute vector potential Ay ---
#     Ay = general_compute_Ay(x_range, y_range, Bx_data, By_data)

#     # --- Smooth Ay ---
#     Ay_smooth = gaussian_filter(Ay, sigma=sigma)

#     # --- Compute gradients and Hessian ---
#     dAydx = np.gradient(Ay_smooth, dx, axis=1)
#     dAydy = np.gradient(Ay_smooth, dy, axis=0)
#     Axx = np.gradient(dAydx, dx, axis=1)
#     Ayy = np.gradient(dAydy, dy, axis=0)
#     Axy = np.gradient(dAydx, dy, axis=0)

#     # --- Critical points where grad ~ 0 ---
#     grad_mag = np.sqrt(dAydx**2 + dAydy**2)
#     tol = grad_tol_factor * np.mean(grad_mag)
#     mask_crit = grad_mag < tol

#     # --- Eigenvalues for classification ---
#     trace = Axx + Ayy
#     det = Axx * Ayy - Axy**2
#     disc = np.sqrt(np.maximum(trace**2 - 4 * det, 0))
#     lambda1 = 0.5 * (trace + disc)
#     lambda2 = 0.5 * (trace - disc)

#     X_mask = mask_crit & (lambda1 * lambda2 < 0)  # saddle (X-point)
#     O_mask = mask_crit & (lambda1 * lambda2 > 0)  # extremum (O-point)

#     # --- Cluster detections ---
#     labeled_X, num_X = label(X_mask)
#     labeled_O, num_O = label(O_mask)

#     def cluster_points(mask, labeled, num, x_range, y_range):
#         coords = center_of_mass(mask, labeled, range(1, num + 1))
#         xs = np.array([x_range[int(c[1])] for c in coords])
#         ys = np.array([y_range[int(c[0])] for c in coords])
#         ix = np.array([int(c[1]) for c in coords])
#         iy = np.array([int(c[0]) for c in coords])
#         return xs, ys, ix, iy

#     xX, yX, ix_X, iy_X = cluster_points(X_mask, labeled_X, num_X, x_range, y_range)
#     xO, yO, ix_O, iy_O = cluster_points(O_mask, labeled_O, num_O, x_range, y_range)

#     print(f"[Frame {frame}] Found {len(xX)} X-points and {len(xO)} O-points.")
#     return xX, yX, ix_X, iy_X, xO, yO, ix_O, iy_O, Ay_smooth, x_range, y_range

##############################################################################
######### To detect the global minimum and based on the value of     #########
######### |Ay(X) - Ay_global_min|, screen the x-point obtained from  #########
######### previous function.                                         #########
##############################################################################

# def filter_x_points_by_Ay(Ay_smooth, x_range, y_range, xX, yX, tol_Ay=0.5):
#     """
#     Filter X-points based on their magnetic potential Ay value relative to the global minimum.

#     Parameters
#     ----------
#     Ay_smooth : 2D array
#         Smoothed magnetic flux function.
#     x_range, y_range : 1D arrays
#         Grid coordinates.
#     xX, yX : 1D arrays
#         Detected X-point coordinates (unfiltered).
#     tol_Ay : float, optional
#         Threshold for |Ay(X) - Ay_global_min|. X-points with smaller differences are discarded.

#     Returns
#     -------
#     xX_filtered, yX_filtered : 1D arrays
#         Filtered X-point coordinates that pass the Ay threshold.
#     Ay_X_filtered : 1D array
#         Corresponding Ay values at the filtered X-points.
#     xO_global, yO_global, Ay_global_min : floats
#         Coordinates and value of the global O-point (global minimum of Ay).
#     """
#     # --- Step 1: Global minimum (O-point) ---
#     Ay_global_min = np.nanmin(Ay_smooth)
#     idx_min = np.unravel_index(np.nanargmin(Ay_smooth), Ay_smooth.shape)
#     yO_global, xO_global = y_range[idx_min[0]], x_range[idx_min[1]]
#     # print(f"Global O-point at (x={xO_global:.3f}, y={yO_global:.3f}), Ay_min={Ay_global_min:.3f}")

#     # --- Step 2: Evaluate Ay at X-points ---
#     Ay_at_X = np.array([
#         Ay_smooth[np.argmin(np.abs(y_range - y)), np.argmin(np.abs(x_range - x))]
#         for x, y in zip(xX, yX)
#     ])

#     # --- Step 3: Filter based on Ay difference ---
#     mask_good_X = np.abs(Ay_at_X - Ay_global_min) >= tol_Ay
#     xX_filtered = xX[mask_good_X]
#     yX_filtered = yX[mask_good_X]
#     Ay_X_filtered = Ay_at_X[mask_good_X]

#     # print(f"Retained {len(xX_filtered)} of {len(xX)} X-points (tol_Ay = {tol_Ay}).")

#     return xX_filtered, yX_filtered, Ay_X_filtered, xO_global, yO_global, Ay_global_min


def filter_x_points_by_Ay(Ay_smooth, x_range, y_range, xX, yX, xO, yO, tol_Ay=0.5, tol_pos=1e-4):
    """
    Filter X-points based on their magnetic potential Ay value relative to the global minimum.

    Parameters
    ----------
    Ay_smooth : 2D array
        Smoothed magnetic flux function.
    x_range, y_range : 1D arrays
        Grid coordinates.
    xX, yX : 1D arrays
        Detected X-point coordinates (unfiltered).
    xO, yO : 1D arrays
        Detected O-point coordinates (from detect_xo_points).
    tol_Ay : float, optional
        Threshold for |Ay(X) - Ay_global_min|. X-points with smaller differences are discarded.
    tol_pos : float, optional
        Spatial tolerance to check if global O-point is among detected O-points.

    Returns
    -------
    xX_filtered, yX_filtered : 1D arrays
        Filtered X-point coordinates that pass the Ay threshold.
    Ay_X_filtered : 1D array
        Corresponding Ay values at the filtered X-points.
    xO_global, yO_global, Ay_global_min : floats
        Coordinates and value of the global O-point (global minimum of Ay).
    """
    # --- Step 1: Global minimum (O-point) ---
    Ay_global_min = np.nanmin(Ay_smooth)
    idx_min = np.unravel_index(np.nanargmin(Ay_smooth), Ay_smooth.shape)
    yO_global, xO_global = y_range[idx_min[0]], x_range[idx_min[1]]

    # --- Step 1a: Check if this global minimum O-point is among detected O-points ---
    if len(xO) > 0:
        dist = np.sqrt((xO - xO_global)**2 + (yO - yO_global)**2)
        if np.any(dist < tol_pos):
            print(f"✅ Global O-point (x={xO_global:.3f}, y={yO_global:.3f}) found among detected O-points.")
        else:
            print(f"⚠️ Global O-point (x={xO_global:.3f}, y={yO_global:.3f}) NOT found among detected O-points.")
    else:
        print("⚠️ No O-points detected to compare with the global minimum.")

    # --- Step 2: Evaluate Ay at X-points ---
    Ay_at_X = np.array([
        Ay_smooth[np.argmin(np.abs(y_range - y)), np.argmin(np.abs(x_range - x))]
        for x, y in zip(xX, yX)
    ])

    # --- Step 3: Filter based on Ay difference ---
    mask_good_X = np.abs(Ay_at_X - Ay_global_min) >= tol_Ay
    xX_filtered = xX[mask_good_X]
    yX_filtered = yX[mask_good_X]
    Ay_X_filtered = Ay_at_X[mask_good_X]

    return xX_filtered, yX_filtered, Ay_X_filtered, xO_global, yO_global, Ay_global_min

    
# import numpy as np

# def filter_x_points_by_Ay(Ay_smooth, x_range, y_range, xX, yX, ix_X, iy_X, tol_Ay=0.5):
#     """
#     Filter X-points based on their magnetic potential Ay value relative to the global minimum.
#     Keeps the corresponding grid indices aligned with the filtered points.

#     Parameters
#     ----------
#     Ay_smooth : 2D array
#         Smoothed magnetic flux function.
#     x_range, y_range : 1D arrays
#         Grid coordinates.
#     xX, yX : 1D arrays
#         Detected X-point coordinates (unfiltered).
#     ix_X, iy_X : 1D integer arrays
#         Grid indices corresponding to the X-points.
#     tol_Ay : float, optional
#         Threshold for |Ay(X) - Ay_global_min|. X-points with smaller differences are discarded.

#     Returns
#     -------
#     xX_filtered, yX_filtered : 1D arrays
#         Filtered X-point coordinates that pass the Ay threshold.
#     ix_X_filtered, iy_X_filtered : 1D integer arrays
#         Corresponding filtered grid indices.
#     Ay_X_filtered : 1D array
#         Corresponding Ay values at the filtered X-points.
#     xO_global, yO_global, Ay_global_min : floats
#         Coordinates and value of the global O-point (global minimum of Ay).
#     """

#     # --- Step 1: Global minimum (O-point) ---
#     Ay_global_min = np.nanmin(Ay_smooth)
#     idx_min = np.unravel_index(np.nanargmin(Ay_smooth), Ay_smooth.shape)
#     iy_min, ix_min = idx_min
#     yO_global, xO_global = y_range[iy_min], x_range[ix_min]

#     print(f"Global O-point at (x={xO_global:.3f}, y={yO_global:.3f}), Ay_min={Ay_global_min:.3f}")

#     # --- Step 2: Evaluate Ay at detected X-points (use indices directly) ---
#     Ay_at_X = Ay_smooth[iy_X, ix_X]

#     # --- Step 3: Filter based on Ay difference ---
#     mask_good_X = np.abs(Ay_at_X - Ay_global_min) >= tol_Ay

#     xX_filtered = xX[mask_good_X]
#     yX_filtered = yX[mask_good_X]
#     ix_X_filtered = ix_X[mask_good_X]
#     iy_X_filtered = iy_X[mask_good_X]
#     Ay_X_filtered = Ay_at_X[mask_good_X]

#     print(f"Retained {len(xX_filtered)} of {len(xX)} X-points (tol_Ay = {tol_Ay}).")

#     return (
#         xX_filtered, yX_filtered, ix_X_filtered, iy_X_filtered,
#         Ay_X_filtered, xO_global, yO_global, Ay_global_min
#     )


###########################################################################################################
###===                            Detecting the O-points. It's a bit tricky                          ===###
###########################################################################################################

# ================================================================
# --- 1. Periodic O-point search between adjacent X-points ---
# ================================================================
# def find_periodic_o_points(Ay_smooth, xX_filtered, yX_filtered, x_range, y_range, buffer_cells=5):
#     """
#     Find O-points between adjacent X-points along periodic y-direction.

#     Parameters
#     ----------
#     Ay_smooth : 2D array
#         Smoothed magnetic flux function.
#     xX_filtered, yX_filtered : 1D arrays
#         Filtered X-point coordinates.
#     x_range, y_range : 1D arrays
#         Grid coordinate arrays.
#     buffer_cells : int
#         Number of grid cells as buffer region when forming local subdomains.

#     Returns
#     -------
#     xO_final, yO_final : 1D arrays
#         Detected O-points (including periodic wrap).
#     """
#     # sort X-points by y (periodic direction)
#     sort_idx = np.argsort(yX_filtered)
#     xX_sorted = xX_filtered[sort_idx]
#     yX_sorted = yX_filtered[sort_idx]

#     O_candidates = []

#     def wrap_y(y):
#         """Wrap y into periodic domain range."""
#         y_span = y_range[-1] - y_range[0]
#         if y < y_range[0]:
#             y += y_span
#         elif y > y_range[-1]:
#             y -= y_span
#         return y

#     # loop over X-point pairs
#     for i in range(len(xX_sorted)):
#         j = (i + 1) % len(xX_sorted)
#         x1, x2 = xX_sorted[i], xX_sorted[j]
#         y1, y2 = yX_sorted[i], yX_sorted[j]

#         # detect if pair crosses periodic boundary
#         y_span = y_range[-1] - y_range[0]
#         crosses = abs(y2 - y1) > 0.5 * y_span

#         # bounding box
#         dx_buffer = buffer_cells * (x_range[1] - x_range[0])
#         dy_buffer = buffer_cells * (y_range[1] - y_range[0])
#         x_min_box = max(min(x1, x2) - dx_buffer, x_range[0])
#         x_max_box = min(max(x1, x2) + dx_buffer, x_range[-1])

#         if crosses:
#             # handle wrapping: stack top & bottom
#             if y1 > y2:
#                 y2 += y_span
#             iy_min = np.searchsorted(y_range, min(y1, y2) - dy_buffer)
#             iy_max = np.searchsorted(y_range, max(y1, y2) + dy_buffer)
#             # tile Ay for periodic continuity
#             Ay_ext = np.vstack((Ay_smooth, Ay_smooth))
#             y_ext = np.concatenate((y_range, y_range + y_span))
#             Ay_sub = Ay_ext[iy_min:iy_max, :]
#             y_sub = y_ext[iy_min:iy_max]
#             ix_min = np.searchsorted(x_range, x_min_box)
#             ix_max = np.searchsorted(x_range, x_max_box)
#             Ay_sub = Ay_sub[:, ix_min:ix_max]
#             x_sub = x_range[ix_min:ix_max]
#         else:
#             y_min_box = max(min(y1, y2) - dy_buffer, y_range[0])
#             y_max_box = min(max(y1, y2) + dy_buffer, y_range[-1])
#             iy_min = np.searchsorted(y_range, y_min_box)
#             iy_max = np.searchsorted(y_range, y_max_box)
#             ix_min = np.searchsorted(x_range, x_min_box)
#             ix_max = np.searchsorted(x_range, x_max_box)
#             Ay_sub = Ay_smooth[iy_min:iy_max, ix_min:ix_max]
#             x_sub, y_sub = x_range[ix_min:ix_max], y_range[iy_min:iy_max]

#         if Ay_sub.size == 0:
#             continue

#         # find local 2D minimum (O-point)
#         iy_local, ix_local = np.unravel_index(np.argmin(Ay_sub), Ay_sub.shape)
#         xO_local, yO_local = x_sub[ix_local], y_sub[iy_local]
#         yO_local = wrap_y(yO_local)
#         O_candidates.append((xO_local, yO_local))

#     if len(O_candidates) == 0:
#         return np.array([]), np.array([])

#     xO_final = np.array([o[0] for o in O_candidates])
#     yO_final = np.array([o[1] for o in O_candidates])

#     print(f"Found {len(xO_final)} O-points (periodic pairs).")
#     return xO_final, yO_final


def find_periodic_o_points(Ay_smooth, xX_filtered, yX_filtered, x_range, y_range, buffer_cells=5):
    """
    Find O-points between adjacent X-points along periodic y-direction.
    Handles correctly the case where adjacent X-points are across the periodic boundary.

    Parameters
    ----------
    Ay_smooth : 2D array
        Smoothed magnetic flux function.
    xX_filtered, yX_filtered : 1D arrays
        Filtered X-point coordinates.
    x_range, y_range : 1D arrays
        Grid coordinate arrays.
    buffer_cells : int
        Number of grid cells as buffer region when forming local subdomains.

    Returns
    -------
    xO_final, yO_final : 1D arrays
        Detected O-points (including periodic wrap).
    """
    if len(xX_filtered) < 2:
        print("Not enough X-points for O-point detection.")
        return np.array([]), np.array([])

    sort_idx = np.argsort(yX_filtered)
    xX_sorted = xX_filtered[sort_idx]
    yX_sorted = yX_filtered[sort_idx]
    O_candidates = []

    def wrap_y(y):
        """Wrap y into periodic domain range."""
        y_span = y_range[-1] - y_range[0]
        while y < y_range[0]:
            y += y_span
        while y > y_range[-1]:
            y -= y_span
        return y

    y_span = y_range[-1] - y_range[0]
    dx_buffer = buffer_cells * (x_range[1] - x_range[0])
    dy_buffer = buffer_cells * (y_range[1] - y_range[0])

    for i in range(len(xX_sorted)):
        j = (i + 1) % len(xX_sorted)
        x1, x2 = xX_sorted[i], xX_sorted[j]
        y1, y2 = yX_sorted[i], yX_sorted[j]
        crosses = abs(y2 - y1) > 0.5 * y_span

        # compute bounding boxes
        x_min_box = max(min(x1, x2) - dx_buffer, x_range[0])
        x_max_box = min(max(x1, x2) + dx_buffer, x_range[-1])

        # --- Case 1: normal (no crossing)
        if not crosses:
            y_min_box = max(min(y1, y2) - dy_buffer, y_range[0])
            y_max_box = min(max(y1, y2) + dy_buffer, y_range[-1])
            ix_min = np.searchsorted(x_range, x_min_box)
            ix_max = np.searchsorted(x_range, x_max_box)
            iy_min = np.searchsorted(y_range, y_min_box)
            iy_max = np.searchsorted(y_range, y_max_box)
            Ay_sub = Ay_smooth[iy_min:iy_max, ix_min:ix_max]
            x_sub, y_sub = x_range[ix_min:ix_max], y_range[iy_min:iy_max]
            if Ay_sub.size == 0:
                continue
            iy_local, ix_local = np.unravel_index(np.argmin(Ay_sub), Ay_sub.shape)
            xO_local, yO_local = x_sub[ix_local], y_sub[iy_local]
            yO_local = wrap_y(yO_local)
            O_candidates.append((xO_local, yO_local))
            continue

        # --- Case 2: periodic crossing (evaluate both boundary regions)
        # region near +y boundary
        iy_min_top = np.searchsorted(y_range, y1) if y1 < y2 else np.searchsorted(y_range, y2)
        Ay_top = Ay_smooth[iy_min_top:, :]
        y_top = y_range[iy_min_top:]
        # region near -y boundary
        iy_max_bot = np.searchsorted(y_range, y2) if y2 < y1 else np.searchsorted(y_range, y1)
        Ay_bot = Ay_smooth[:iy_max_bot, :]
        y_bot = y_range[:iy_max_bot]

        # Subset x-range (same for both)
        ix_min = np.searchsorted(x_range, x_min_box)
        ix_max = np.searchsorted(x_range, x_max_box)
        Ay_top = Ay_top[:, ix_min:ix_max]
        Ay_bot = Ay_bot[:, ix_min:ix_max]
        x_sub = x_range[ix_min:ix_max]

        # Find minima in both
        min_vals = []
        if Ay_top.size > 0:
            iy_t, ix_t = np.unravel_index(np.argmin(Ay_top), Ay_top.shape)
            min_vals.append(('top', Ay_top[iy_t, ix_t], x_sub[ix_t], y_top[iy_t]))
        if Ay_bot.size > 0:
            iy_b, ix_b = np.unravel_index(np.argmin(Ay_bot), Ay_bot.shape)
            min_vals.append(('bot', Ay_bot[iy_b, ix_b], x_sub[ix_b], y_bot[iy_b]))

        if len(min_vals) == 0:
            continue

        # Choose smaller Ay, or higher decimal precision if equal
        if len(min_vals) == 1:
            chosen = min_vals[0]
        else:
            top_val, bot_val = min_vals[0][1], min_vals[1][1]
            if np.isclose(top_val, bot_val, atol=1e-10):
                # equal to machine precision, discard one to avoid duplicates
                continue
            chosen = min(min_vals, key=lambda t: (t[1], round(t[1], 10)))

        _, Ay_min_val, x_sel, y_sel = chosen
        y_sel = wrap_y(y_sel)
        O_candidates.append((x_sel, y_sel))

    if len(O_candidates) == 0:
        return np.array([]), np.array([])

    xO_final = np.array([o[0] for o in O_candidates])
    yO_final = np.array([o[1] for o in O_candidates])
    # print(f"Found {len(xO_final)} O-points (periodic, with dual-boundary check).")
    return xO_final, yO_final

####================================================================
####--- 2. Boundary-region O-point detection using square region ---
####================================================================
def find_boundary_o_point(Ay_smooth, xX_filtered, yX_filtered, x_range, y_range):
    """
    Find O-point that lies across the periodic y-boundaries using
    a 2D square search region around the top and bottom boundary X-points.
    """
    def phys_to_index(x, y):
        ix = np.argmin(np.abs(x_range - x))
        iy = np.argmin(np.abs(y_range - y))
        return ix, iy

    def find_min_in_square(Ay, ix_center, iy_start, iy_end):
        ny, nx = Ay.shape
        height = abs(iy_end - iy_start)
        half_width = height
        ix_min = max(0, ix_center - half_width)
        ix_max = min(nx - 1, ix_center + half_width)
        iy_min = min(iy_start, iy_end)
        iy_max = max(iy_start, iy_end)
        sub = Ay[iy_min:iy_max + 1, ix_min:ix_max + 1]
        if sub.size == 0:
            return None, None, None
        local_idx = np.unravel_index(np.argmin(sub), sub.shape)
        iy_min_local = iy_min + local_idx[0]
        ix_min_local = ix_min + local_idx[1]
        return iy_min_local, ix_min_local, float(sub[local_idx])

    if len(xX_filtered) == 0:
        return np.array([]), np.array([])

    X_yvals = np.array(yX_filtered)
    idx_pos = np.argmax(X_yvals)
    idx_neg = np.argmin(X_yvals)
    x_pos, y_pos = xX_filtered[idx_pos], yX_filtered[idx_pos]
    x_neg, y_neg = xX_filtered[idx_neg], yX_filtered[idx_neg]
    ix_pos, iy_pos = phys_to_index(x_pos, y_pos)
    ix_neg, iy_neg = phys_to_index(x_neg, y_neg)

    # Positive boundary
    iy_min_pos, ix_min_pos, Ay_min_pos = find_min_in_square(Ay_smooth, ix_pos, iy_pos, Ay_smooth.shape[0]-1)
    xO_pos_phys = x_range[ix_min_pos] if ix_min_pos is not None else None
    yO_pos_phys = y_range[iy_min_pos] if iy_min_pos is not None else None

    # Negative boundary
    iy_min_neg, ix_min_neg, Ay_min_neg = find_min_in_square(Ay_smooth, ix_neg, 0, iy_neg)
    xO_neg_phys = x_range[ix_min_neg] if ix_min_neg is not None else None
    yO_neg_phys = y_range[iy_min_neg] if iy_min_neg is not None else None

    # Choose smaller Ay
    boundary_o = None
    if (Ay_min_pos is not None) and (Ay_min_neg is not None):
        if Ay_min_pos <= Ay_min_neg:
            boundary_o = (xO_pos_phys, yO_pos_phys)
        else:
            boundary_o = (xO_neg_phys, yO_neg_phys)
    elif Ay_min_pos is not None:
        boundary_o = (xO_pos_phys, yO_pos_phys)
    elif Ay_min_neg is not None:
        boundary_o = (xO_neg_phys, yO_neg_phys)

    if boundary_o is None:
        return np.array([]), np.array([])
    else:
        return np.array([boundary_o[0]]), np.array([boundary_o[1]])



# ================================================================
# --- 3. Deduplication and cleanup ---
# ================================================================
def deduplicate_points(xX, yX, xO, yO, overlap_tol, keep='X'):
    """
    Remove overlapping X and O points.

    Parameters
    ----------
    xX, yX, xO, yO : 1D arrays
        Coordinates of X- and O-points.
    overlap_tol : float
        Tolerance distance for considering overlap.
    keep : {'X', 'O'}
        Which type to prefer when overlap occurs.

    Returns
    -------
    x_points, o_points : arrays (N×2)
        Cleaned final sets of X- and O-points.
    """
    xX_arr, yX_arr = np.array(xX), np.array(yX)
    xO_arr, yO_arr = np.array(xO), np.array(yO)

    # remove O overlapping with X
    keep_O = np.ones(len(xO_arr), dtype=bool)
    for i, (xo, yo) in enumerate(zip(xO_arr, yO_arr)):
        d_to_X = np.hypot(xX_arr - xo, yX_arr - yo)
        if np.any(d_to_X <= overlap_tol):
            if keep == 'X':
                keep_O[i] = False
    xO_arr, yO_arr = xO_arr[keep_O], yO_arr[keep_O]

    # remove duplicate Xs
    keep_X = np.ones(len(xX_arr), dtype=bool)
    for i in range(len(xX_arr)):
        if not keep_X[i]:
            continue
        for j in range(i+1, len(xX_arr)):
            if np.hypot(xX_arr[i]-xX_arr[j], yX_arr[i]-yX_arr[j]) <= overlap_tol:
                keep_X[j] = False
    xX_arr, yX_arr = xX_arr[keep_X], yX_arr[keep_X]

    # remove duplicate Os
    keep_O = np.ones(len(xO_arr), dtype=bool)
    for i in range(len(xO_arr)):
        if not keep_O[i]:
            continue
        for j in range(i+1, len(xO_arr)):
            if np.hypot(xO_arr[i]-xO_arr[j], yO_arr[i]-yO_arr[j]) <= overlap_tol:
                keep_O[j] = False
    xO_arr, yO_arr = xO_arr[keep_O], yO_arr[keep_O]

    x_points = np.vstack((xX_arr, yX_arr)).T
    o_points = np.vstack((xO_arr, yO_arr)).T
    return x_points, o_points



def deduplicate_points_topo(
    xX, yX, xO, yO, Ay_smooth, x_range, y_range,
    Ay_global_min, square_size=1.0, periodic_y=True
):
    """
    Robust deduplication of X and O points using topological consistency
    and flux-based filtering.

    Parameters
    ----------
    xX, yX, xO, yO : 1D arrays
        Coordinates of detected X- and O-points.
    Ay_smooth : 2D array
        Smoothed magnetic flux function.
    x_range, y_range : 1D arrays
        Simulation coordinate axes.
    Ay_global_min : float
        Global minimum of Ay (reference for reconnection layer).
    square_size : float
        Half-length of square (in same units as x/y) for local neighborhood check.
    periodic_y : bool
        Whether y-direction is periodic.

    Returns
    -------
    xX_clean, yX_clean, xO_clean, yO_clean : 1D arrays
        Cleaned lists of X and O points.
    """

    # Combine and sort all critical points
    types = (['X'] * len(xX)) + (['O'] * len(xO))
    x_all = np.concatenate([xX, xO])
    y_all = np.concatenate([yX, yO])

    # Helper to wrap y periodicity
    y_span = y_range[-1] - y_range[0]
    if periodic_y:
        y_all_wrapped = np.mod(y_all - y_range[0], y_span) + y_range[0]
    else:
        y_all_wrapped = y_all.copy()

    # Sort by y (periodic sense)
    order = np.argsort(y_all_wrapped)
    x_all, y_all, y_all_wrapped, types = (
        x_all[order], y_all[order], y_all_wrapped[order], np.array(types)[order]
    )

    # Helper: sample Ay value at a point
    def Ay_at_point(xp, yp):
        ix = np.argmin(np.abs(x_range - xp))
        iy = np.argmin(np.abs(y_range - yp))
        return Ay_smooth[iy, ix]

    # Build Ay arrays for convenience
    Ay_vals = np.array([Ay_at_point(xp, yp) for xp, yp in zip(x_all, y_all)])

    # Output masks (True = keep)
    keep_mask = np.ones(len(x_all), dtype=bool)

    # Iterate through points, check for nearby duplicates
    for i in range(len(x_all)):
        if not keep_mask[i]:
            continue

        xi, yi, ti = x_all[i], y_all_wrapped[i], types[i]

        # find all points within square neighborhood
        dx = np.abs(x_all - xi)
        dy = np.abs(y_all_wrapped - yi)
        # handle periodic wrap in y
        if periodic_y:
            dy = np.minimum(dy, y_span - dy)

        nearby = np.where((dx <= square_size) & (dy <= square_size))[0]
        if len(nearby) == 1:
            continue  # isolated, keep

        # Determine nearest neighbors along y
        left = (i - 1) % len(x_all)
        right = (i + 1) % len(x_all)
        left_type, right_type = types[left], types[right]

        # Case A: neighbors are O-points → only one X should remain
        if left_type == 'O' and right_type == 'O':
            xs_inside = [j for j in nearby if types[j] == 'X']
            if len(xs_inside) > 1:
                # keep the X with largest |Ay - Ay_global_min|
                Ay_diff = np.abs(Ay_vals[xs_inside] - Ay_global_min)
                best_idx = xs_inside[np.argmax(Ay_diff)]
                for j in xs_inside:
                    if j != best_idx:
                        keep_mask[j] = False
            # remove all O's in this box
            for j in nearby:
                if types[j] == 'O':
                    keep_mask[j] = False

        # Case B: neighbors are X-points → only one O should remain
        elif left_type == 'X' and right_type == 'X':
            os_inside = [j for j in nearby if types[j] == 'O']
            if len(os_inside) > 1:
                # keep O with smallest Ay
                best_idx = os_inside[np.argmin(Ay_vals[os_inside])]
                for j in os_inside:
                    if j != best_idx:
                        keep_mask[j] = False
            # remove all X's in this box
            for j in nearby:
                if types[j] == 'X':
                    keep_mask[j] = False

        # Other combinations (mixed or boundary): keep only strongest distinct one
        else:
            # pick the single point with extreme Ay difference
            Ay_diff = np.abs(Ay_vals[nearby] - Ay_global_min)
            best_idx = nearby[np.argmax(Ay_diff)]
            for j in nearby:
                if j != best_idx:
                    keep_mask[j] = False

    # Filter
    x_clean = x_all[keep_mask]
    y_clean = y_all[keep_mask]
    types_clean = types[keep_mask]

    # Re-split into X and O arrays
    xX_clean = x_clean[types_clean == 'X']
    yX_clean = y_clean[types_clean == 'X']
    xO_clean = x_clean[types_clean == 'O']
    yO_clean = y_clean[types_clean == 'O']

    # ---- Check topological alternation (X–O–X–O...) ----
    if len(xX_clean) > 0 and len(xO_clean) > 0:
        all_types = np.array(
            ['X' if t == 'X' else 'O' for t in types_clean]
        )
        # Check sequence consistency
        seq_good = True
        for i in range(len(all_types) - 1):
            if all_types[i] == all_types[i + 1]:
                seq_good = False
                break
        if not seq_good:
            print("[Warning] Alternation violated after cleanup — manual inspection suggested.")

    return xX_clean, yX_clean, xO_clean, yO_clean

##################################################################################################
##################################################################################################
##################################################################################################



def filter_and_pair_xo_points(Ay_smooth, x_range, y_range,
                              xX, yX, xO, yO,
                              tol_Ay=0.5,
                              use_gradient_tie_break=True):
    """
    Filter X-points and O-points and pair them so they alternate along periodic y.

    Parameters
    ----------
    Ay_smooth : 2D ndarray
        Smoothed flux function (Ay[y_index, x_index]).
    x_range, y_range : 1D ndarray
        Grid coordinates (must match axes of Ay_smooth).
    xX, yX, xO, yO : 1D ndarrays
        Raw detected X- and O-point coordinates (in physical coords).
    tol_Ay : float
        Threshold for filtering X by |Ay(X) - Ay_global_min| >= tol_Ay
    use_gradient_tie_break : bool
        If True, when two O-points have equal Ay (within eps) choose the one with
        smaller |grad Ay| (more extremal).

    Returns
    -------
    xX_paired, yX_paired, xO_paired, yO_paired : ndarrays
        Paired/filtered coordinates. They are ordered by increasing y of the X points
        (periodic wrap handled) and have equal length.
    info : dict
        Diagnostic info: counts, global O coords/value, removed counts, etc.
    """
    # --- Basic checks ---
    if len(xX) == 0 or len(xO) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), {
            'reason': 'no_x_or_o_detected',
            'nX_raw': len(xX), 'nO_raw': len(xO)
        }

    # --- Helpers to map physical coords to grid indices ---
    def idx_of_coord(xarr, xval):
        # returns nearest index
        return int(np.argmin(np.abs(xarr - xval)))

    # Domain periodic length in y (assumes y_range sorted increasing)
    y_min = y_range[0]
    dy = y_range[1] - y_range[0]
    Ly = (y_range[-1] - y_range[0]) + dy  # total periodic length

    # --- Global O min (ensure present among O candidates) ---
    idx_min = np.unravel_index(np.nanargmin(Ay_smooth), Ay_smooth.shape)
    yO_global_idx, xO_global_idx = idx_min[0], idx_min[1]
    Ay_global_min = Ay_smooth[yO_global_idx, xO_global_idx]
    xO_global = x_range[xO_global_idx]
    yO_global = y_range[yO_global_idx]

    # If global O not in the O-list (within one cell), append it
    # Build arrays of indices for existing O candidates
    O_indices = np.array([[idx_of_coord(x_range, xo), idx_of_coord(y_range, yo)] for xo, yo in zip(xO, yO)])
    # check if any candidate coincides with global O index
    if not np.any((O_indices[:,0] == xO_global_idx) & (O_indices[:,1] == yO_global_idx)):
        # append
        xO = np.append(xO, xO_global)
        yO = np.append(yO, yO_global)

    # --- Filter X by Ay distance from global O ---
    # compute Ay at X points
    X_idx = np.array([[idx_of_coord(x_range, xx), idx_of_coord(y_range, yy)] for xx, yy in zip(xX, yX)])
    Ay_at_X = np.array([Ay_smooth[yidx, xidx] for xidx, yidx in X_idx])  # note ordering
    # keep X where |Ay(X) - Ay_global_min| >= tol_Ay
    mask_keepX = np.abs(Ay_at_X - Ay_global_min) >= tol_Ay
    xX_filt = np.array(xX)[mask_keepX]
    yX_filt = np.array(yX)[mask_keepX]

    # If none left, relax criterion slightly (avoid losing all)
    if len(xX_filt) == 0:
        # relax by half
        mask_keepX = np.abs(Ay_at_X - Ay_global_min) >= (tol_Ay * 0.5)
        xX_filt = np.array(xX)[mask_keepX]
        yX_filt = np.array(yX)[mask_keepX]

    # --- If still none, return diagnostic ---
    if len(xX_filt) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), {
            'reason': 'no_x_after_filter',
            'nX_raw': len(xX), 'tol_Ay': tol_Ay
        }

    # --- Sort filtered X by y in increasing order and map to [0, Ly) interval ---
    # Normalize y to [0, Ly)
    def wrap_y(y):
        y0 = y - y_min
        y0 = np.mod(y0, Ly)
        return y0

    xX_filt = np.array(xX_filt)
    yX_filt = np.array(yX_filt)
    yX_wrapped = wrap_y(yX_filt)
    sort_idx = np.argsort(yX_wrapped)
    xX_sorted = xX_filt[sort_idx]
    yX_sorted = yX_filt[sort_idx]
    yX_wrapped_sorted = yX_wrapped[sort_idx]

    # --- Prepare O candidates arrays + Ay and gradient if needed ---
    xO_arr = np.array(xO)
    yO_arr = np.array(yO)
    # compute Ay for each O candidate
    O_idx = np.array([[idx_of_coord(x_range, xo), idx_of_coord(y_range, yo)] for xo, yo in zip(xO_arr, yO_arr)])
    Ay_at_O = np.array([Ay_smooth[yidx, xidx] for xidx, yidx in O_idx])
    # gradient magnitude for tie-breaking, if needed
    if use_gradient_tie_break:
        # central differences on Ay_smooth (grid: Ay[y,x])
        # compute grad magnitude at each O-grid index
        # use numpy.gradient
        gy, gx = np.gradient(Ay_smooth, y_range, x_range, edge_order=2)
        grad_mag_at_O = np.array([np.hypot( gx[xidx, yidx] if False else gx[yidx, xidx],
                                            gy[xidx, yidx] if False else gy[yidx, xidx]) 
                                  for xidx, yidx in O_idx])
        # Note: ordering in gradient indexing above uses [y,x], adapt accordingly
        # Simpler: index as [yidx, xidx] for arrays shaped [ny,nx]
        grad_mag_at_O = np.array([np.hypot(gx[yidx, xidx], gy[yidx, xidx]) for xidx, yidx in O_idx])
    else:
        grad_mag_at_O = np.full(len(Ay_at_O), np.inf)

    # --- For each adjacent pair of X (with wrap-around) find O candidates in interval ---
    def y_in_interval(y_cand_wrapped, y_start_wrapped, y_end_wrapped):
        """
        Check whether y_cand_wrapped lies in the interval (y_start_wrapped, y_end_wrapped]
        moving forward in y modulo Ly. y_* are in [0,Ly).
        """
        if y_end_wrapped > y_start_wrapped:
            return (y_cand_wrapped > y_start_wrapped) & (y_cand_wrapped <= y_end_wrapped)
        else:
            # wrap-around case (end < start)
            return (y_cand_wrapped > y_start_wrapped) | (y_cand_wrapped <= y_end_wrapped)

    nX = len(xX_sorted)
    xO_selected = []
    yO_selected = []
    xX_selected = []
    yX_selected = []

    # Precompute wrapped y for O candidates
    yO_wrapped = wrap_y(yO_arr)

    for i in range(nX):
        j = (i + 1) % nX  # next index with wrap
        y_start_wr = yX_wrapped_sorted[i]
        y_end_wr = yX_wrapped_sorted[j]
        # find O candidates in this y-interval
        in_interval_mask = y_in_interval(yO_wrapped, y_start_wr, y_end_wr)
        if not np.any(in_interval_mask):
            # No O in strict interval -> fallback: consider O closer to mid-y between Xs
            # compute mid point in wrapped coords
            if y_end_wr > y_start_wr:
                y_mid = 0.5 * (y_start_wr + y_end_wr)
            else:
                # wrap-around mid
                y_end_wr_adj = y_end_wr + Ly
                y_mid = 0.5 * (y_start_wr + y_end_wr_adj)
                y_mid = np.mod(y_mid, Ly)
            # pick closest O in wrapped distance
            dy_wr = np.abs(np.mod(yO_wrapped - y_mid + Ly/2, Ly) - Ly/2)
            nearest_idx = np.argmin(dy_wr)
            in_interval_idx = [nearest_idx]
        else:
            in_interval_idx = np.where(in_interval_mask)[0]

        # among in_interval_idx pick O with minimum Ay (tie-break with grad mag)
        Ay_candidates = Ay_at_O[in_interval_idx]
        minAy = np.min(Ay_candidates)
        # indices within global O-arr
        cand_indices = np.array(in_interval_idx)[np.isclose(Ay_candidates, minAy, atol=1e-12)]
        if len(cand_indices) > 1:
            # tie-break by gradient (choose smaller gradient magnitude => deeper extremum)
            grads = grad_mag_at_O[cand_indices]
            best_local = cand_indices[np.argmin(grads)]
        else:
            best_local = cand_indices[0]

        # store pair
        xO_selected.append(xO_arr[best_local])
        yO_selected.append(yO_arr[best_local])
        xX_selected.append(xX_sorted[i])
        yX_selected.append(yX_sorted[i])

    # convert to arrays
    xX_paired = np.array(xX_selected)
    yX_paired = np.array(yX_selected)
    xO_paired = np.array(xO_selected)
    yO_paired = np.array(yO_selected)

    # --- final consistency: ensure lengths equal, otherwise trim the larger set by proximity ---
    if len(xX_paired) != len(xO_paired):
        # pair by nearest: find minimal matching set
        # compute distance matrix and pick minimal pairing greedily
        NX = len(xX_paired); NO = len(xO_paired)
        X_coords = np.vstack((xX_paired, yX_paired)).T
        O_coords = np.vstack((xO_paired, yO_paired)).T
        # compute periodic dy when computing distances
        def pairwise_dist_matrix(Xc, Oc):
            # Xc: (NX,2), Oc: (NO,2)
            dx = Xc[:, None, 0] - Oc[None, :, 0]
            dy = Xc[:, None, 1] - Oc[None, :, 1]
            # wrap dy
            dy = np.mod(dy - y_min + Ly/2, Ly) - Ly/2
            return np.hypot(dx, dy)
        D = pairwise_dist_matrix(X_coords, O_coords)
        # greedy matching
        X_used = set()
        O_used = set()
        pairs = []
        # flatten sorted by distance
        order = np.dstack(np.unravel_index(np.argsort(D.ravel()), D.shape))[0]
        for xi, oi in order:
            if xi not in X_used and oi not in O_used:
                pairs.append((xi, oi))
                X_used.add(xi); O_used.add(oi)
            if len(X_used) == min(NX, NO) or len(O_used) == min(NX, NO):
                break
        # build new arrays
        pairs = sorted(pairs, key=lambda p: wrap_y(yX_paired[p[0]]))  # order by X y
        xX_paired = np.array([xX_paired[p[0]] for p in pairs])
        yX_paired = np.array([yX_paired[p[0]] for p in pairs])
        xO_paired = np.array([xO_paired[p[1]] for p in pairs])
        yO_paired = np.array([yO_paired[p[1]] for p in pairs])

    info = {
        'nX_raw': len(xX),
        'nO_raw': len(xO),
        'nX_filtered': len(xX_paired),
        'nO_filtered': len(xO_paired),
        'xO_global': xO_global, 'yO_global': yO_global, 'Ay_global_min': Ay_global_min
    }

    return xX_paired, yX_paired, xO_paired, yO_paired, info


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

