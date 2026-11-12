###ticks_gids_clean.py

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
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba

# from scipy.interpolate import interp1d

def set_ticks_and_grids(ax,
                        major_interval_x=200,
                        minor_interval_x=None,
                        major_interval_y=2,
                        minor_interval_y=None,
                        tick_labelsize=30,
                        major_tick_length=10,
                        minor_tick_length=5,
                        major_tick_width=3,
                        minor_tick_width=1.5,
                        major_tick_color='black',
                        minor_tick_color='black',
                        label_color_x = 'black',
                        label_color_y = 'black',
                        major_grid=True,
                        minor_grid=True,
                        major_grid_style='-',
                        minor_grid_style='--',
                        major_grid_width=1.2,
                        minor_grid_width=0.8,
                        major_grid_alpha=0.6,  # Control major grid opacity
                        minor_grid_alpha=0.3,  # Control minor grid opacity
                        grid_color='gray',
			xlog = False,
			ylog = False):
    if minor_interval_x is None:
        minor_interval_x = major_interval_x / 10.0
    if minor_interval_y is None:
        minor_interval_y = major_interval_y / 5.0

    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=major_interval_x))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=minor_interval_x))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=major_interval_y))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=minor_interval_y))

    ax.tick_params(axis='x', which='major', length=major_tick_length,
                   width=major_tick_width, labelsize=tick_labelsize, color=major_tick_color, labelcolor = label_color_x)
    ax.tick_params(axis='y', which='major', length=major_tick_length,
                   width=major_tick_width, labelsize=tick_labelsize, color=major_tick_color, labelcolor = label_color_y)
    ax.tick_params(axis='x', which='minor', length=minor_tick_length,
                   width=minor_tick_width, labelsize=tick_labelsize, color=major_tick_color)
    ax.tick_params(axis='y', which='minor', length=minor_tick_length,
                   width=minor_tick_width, labelsize=tick_labelsize, color=major_tick_color)

    # if major_grid:
    #     ax.grid(True, which='major', linestyle=major_grid_style,
    #             linewidth=major_grid_width, alpha=major_grid_alpha, color=grid_color)
    # if minor_grid:
    #     ax.grid(True, which='minor', linestyle=minor_grid_style,
    #             linewidth=minor_grid_width, alpha=minor_grid_alpha, color=grid_color)

    if major_grid:
        ax.grid(True, which='major', linestyle=major_grid_style,
                linewidth=major_grid_width, color=to_rgba(grid_color, alpha=major_grid_alpha))
    if minor_grid:
        ax.grid(True, which='minor', linestyle=minor_grid_style,
                linewidth=minor_grid_width, color=to_rgba(grid_color, alpha=minor_grid_alpha))

    ax.tick_params(which='both', direction='in', right=True)

    # Optional log scale
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')

def plot_trajectory_line_collection(ax1, frame, x, z, time_array, wp1, cmap, norm, boundary_width=409.6, threshold=0.5):
    points = np.array([x[:frame+1], z[:frame+1]]).T
    segments = []
    colors = []

    for i in range(frame):
        dx = abs(x[i+1] - x[i])
        if dx > threshold * boundary_width:
            continue
        segments.append([[x[i], z[i]], [x[i+1], z[i+1]]])
        colors.append(wp1 * time_array[i])

    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(np.array(colors))
    lc.set_linewidth(3)
    ax1.add_collection(lc)

    return lc

# Helper: get unique linestyle and marker combination
def get_style_marker(index):
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', '^', 's', '*', 'x', 'D', 'P', '+']
    style_combinations = [(ls, mk) for ls in linestyles for mk in markers]
    return style_combinations[index % len(style_combinations)]
'''
# Modified plotting function with marker color matching segment color
def plot_discrete_colored_styled_curve(ax, x, y, levels, color_line, 
                                       linestyle='-', linewidth=2, marker='o', 
                                       markevery=10, label=None):
    # Plot segments with time-colored line
    for i in range(len(x)-1):
        color_idx = np.digitize(x[i], levels) - 1
        color_idx = np.clip(color_idx, 0, len(color_line)-1)
        ax.plot(x[i:i+2], y[i:i+2], color=color_line[color_idx], 
                linestyle=linestyle, linewidth=linewidth)

    # Plot markers with matching color
    for i in range(0, len(x), markevery):
        color_idx = np.digitize(x[i], levels) - 1
        color_idx = np.clip(color_idx, 0, len(color_line)-1)
        ax.plot(x[i], y[i], marker=marker, markersize=6, 
                color=color_line[color_idx], markeredgecolor='k', linestyle='none')

    # Add dummy invisible line for legend
    ax.plot([], [], linestyle=linestyle, marker=marker, 
            color='gray', linewidth=linewidth, label=label)
'''

def plot_discrete_colored_styled_curve(ax, x, y, levels, color_line,
                                       linestyle='-', linewidth=2, marker='o',
                                       markevery=10, label=None,
                                       markersize=6.0, markeredgewidth=1.0):
    """
    Plot a line segmented with color and matching markers.

    Parameters:
    - ax : matplotlib axis
    - x, y : arrays of data points
    - levels : array of level thresholds for coloring
    - color_line : list or colormap of colors
    - linestyle : line style
    - linewidth : line width
    - marker : marker style
    - markevery : interval to place markers
    - label : legend label
    - markersize : size of the markers (float)
    - markeredgewidth : width of the marker edge (float)
    """

    # Plot line segments with color
    for i in range(len(x) - 1):
        color_idx = np.digitize(x[i], levels) - 1
        color_idx = np.clip(color_idx, 0, len(color_line) - 1)
        ax.plot(x[i:i + 2], y[i:i + 2], color=color_line[color_idx],
                linestyle=linestyle, linewidth=linewidth)

    # Plot markers with matching color
    for i in range(0, len(x), markevery):
        color_idx = np.digitize(x[i], levels) - 1
        color_idx = np.clip(color_idx, 0, len(color_line) - 1)
        ax.plot(x[i], y[i], marker=marker, markersize=markersize,
                color=color_line[color_idx], markeredgecolor='k',
                markeredgewidth=markeredgewidth, linestyle='none')

    # Dummy plot for legend
    ax.plot([], [], linestyle=linestyle, marker=marker,
            color='gray', linewidth=linewidth, markersize=markersize,
            markeredgewidth=markeredgewidth, label=label)



def set_spine_linewidth(ax, width=2, spines=['top', 'bottom', 'left', 'right']):
    for spine in spines:
        ax.spines[spine].set_linewidth(width)



def add_colorbar(fig, ax, mappable, levels, location='top', label=r'$\omega_{p} t$',
                 fontsize=20, label_offset=None, label_rotation=None):
    """
    Add a colorbar to a matplotlib figure on any side of a given axis (top, bottom, left, right).

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The main axis to which the colorbar will be attached.
    mappable : The object returned by a plotting function like `contourf`, `pcolormesh`, `LineCollection`, etc.
    levels : list or array
        Tick positions for the colorbar.
    location : str
        One of 'top', 'bottom', 'left', 'right'. Default is 'top'.
    label : str
        Label for the colorbar. Default is '$\omega_{p} t$'.
    fontsize : int
        Font size for ticks and label.
    label_offset : float or None
        Offset for manual label placement (in axes coordinates). Default is auto-handled.
    label_rotation : float or None
        Rotation of label in degrees. Default is auto-handled per location.

    Returns:
    --------
    cbar : matplotlib.colorbar.Colorbar
        The colorbar instance.
    """
    # Remove old colorbar if exists
    if hasattr(add_colorbar, "cbar") and add_colorbar.cbar:
        add_colorbar.cbar.remove()

    divider = make_axes_locatable(ax)

    if location == 'top':
        cax = divider.append_axes('top', size='5%', pad=0.4)
        orientation = 'horizontal'
    elif location == 'bottom':
        cax = divider.append_axes('bottom', size='5%', pad=0.4)
        orientation = 'horizontal'
    elif location == 'left':
        cax = divider.append_axes('left', size='5%', pad=0.4)
        orientation = 'vertical'
    elif location == 'right':
        cax = divider.append_axes('right', size='5%', pad=0.4)
        orientation = 'vertical'
    else:
        raise ValueError("location must be 'top', 'bottom', 'left', or 'right'")

    # Create colorbar
    cbar = fig.colorbar(mappable, cax=cax, ax=ax,
                        orientation=orientation,
                        shrink=0.99, pad=0.01,
                        aspect=18,
                        boundaries=levels,
                        ticks=levels)

    cbar.ax.tick_params(labelsize=fontsize, width=1.5, length=7, direction='in')

    # Remove default label
    cbar.set_label('')

    # Set tick position
    if location in ['top', 'bottom']:
        cax.xaxis.set_ticks_position(location)
        cax.xaxis.set_label_position(location)
        x_pos = -0.06 if location == 'top' else -0.06
        y_pos = 1.1 if location == 'top' else -0.5
        if label_offset is not None:
            y_pos += label_offset
        if label_rotation is None:
            label_rotation = 0

        cbar.ax.text(x_pos, y_pos, label,
                     fontsize=fontsize,
                     ha='left', va='center',
                     transform=cbar.ax.transAxes,
                     rotation=label_rotation)

    elif location in ['left', 'right']:
        cax.yaxis.set_ticks_position(location)
        cax.yaxis.set_label_position(location)
        x_pos = -0.6 if location == 'left' else 1.1
        y_pos = 1.05
        if label_offset is not None:
            x_pos += label_offset
        if label_rotation is None:
            label_rotation = 90

        cbar.ax.text(x_pos, y_pos, label,
                     fontsize=fontsize,
                     ha='center', va='top',
                     transform=cbar.ax.transAxes,
                     rotation=label_rotation)

    add_colorbar.cbar = cbar  # store reference for reuse/removal
    return cbar



def cumulative_gauss_quad(time_points, integrand, rest_mass_energy, ke_data,
                           quad_order=2, interp_kind='cubic'):
    """
    Perform cumulative time integration using Gaussian quadrature of arbitrary order.

    Parameters:
    ----------
    time_points : ndarray of shape (N,)
        Time axis values.
    integrand : ndarray of shape (N, M)
        Integrand values over time for M particles/quantities.
    rest_mass_energy : float
        Normalization factor.
    ke_data : ndarray of shape (M,)
        Initial kinetic energy data.
    quad_order : int, optional
        Order of Gaussian quadrature (default: 2).
    interp_kind : str, optional
        Interpolation kind for integrand values at quadrature points ('linear', 'cubic', etc.).

    Returns:
    -------
    cumulative_integral : ndarray of shape (N, M)
        Cumulative integration result at each time step for all quantities.
    """
    N, M = integrand.shape
    cumulative_integral = np.zeros((N, M))

    # Get Gauss–Legendre nodes and weights for specified order
    xg, wg = leggauss(quad_order)

    # Precompute interpolators (one for each particle/quantity)
    interpolators = [interp1d(time_points, integrand[:, j], kind=interp_kind,
                              bounds_error=False, fill_value="extrapolate")
                     for j in range(M)]

    for i in range(N):
        if i == 0:
            # First timestep: initialize with KE + 1
            cumulative_integral[i, :] = ke_data[0] + 1.0
        else:
            t0, t1 = time_points[i-1], time_points[i]
            dt = (t1 - t0) / 2.0
            tm = (t1 + t0) / 2.0
            t_quad = dt * xg + tm  # map to interval [t0, t1]

            # Interpolate integrand at each Gauss point
            integrand_quad = np.zeros((quad_order, M))
            for j in range(M):
                integrand_quad[:, j] = interpolators[j](t_quad)

            # Apply Gaussian quadrature: sum over Gauss points
            integral = dt * np.dot(wg, integrand_quad)  # shape (M,)
            cumulative_integral[i, :] = cumulative_integral[i-1, :] + integral / rest_mass_energy

    return cumulative_integral



#### Flux function ####
def compute_Ay(x, z, Bx, Bz):
    Nx = len(x)
    Nz = len(z)
    dx = np.diff(x)
    dz = np.diff(z)

    # Output array
    Ay = np.zeros((Nz, Nx))

    # First term: -∫ Bx(x, z') dz' at fixed x
    for ix in range(Nx):
        Ay[1:, ix] = -np.cumsum(0.5 * (Bx[1:, ix] + Bx[:-1, ix]) * dz[:, None].flatten())

    # Second term: +∫ Bz(x', 0) dx'
    Ay[0, 1:] = np.cumsum(0.5 * (Bz[0, 1:] + Bz[0, :-1]) * dx)

    # Broadcast to full 2D
    for iz in range(1, Nz):
        Ay[iz, :] += Ay[0, :]

    return Ay


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




#########################################################################################################
#########                                Modified Ay                                         ############
#########################################################################################################
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

######################################################################################################################
#################                 To make the 4 plots in a row: Jz and Ay                          ###################
######################################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from matplotlib.colors import SymLogNorm
from matplotlib.transforms import Bbox
from fitting_beta import tanh_model, fit_By_tanh_normalized

def plot_row4_jz_with_Ay(
    S, frames, di, B0,
    *,
    min_val=-2.2, max_val=2.2,
    linthresh=1e-3, linscale=1.0,
    cmap="PuOr",
    n_contours=25,
    contour_percentiles=(5, 95),
    figsize=(22, 6),
    wspace=0.02,            # <<< tighten spacing between panels
    aspect="auto",
    ls=15,

    # --- figure margins (these control whitespace a lot) ---
    left=0.06,
    bottom=0.14,            # leave room for shared xlabel
    top=0.93,

    # --- colorbar geometry: independent width/length control ---
    cbar_pad=0.01,          # gap between last panel and colorbar (figure fraction)
    cbar_width=0.018,       # colorbar width (figure fraction)  <<< independent
    cbar_length=0.90,       # colorbar length as fraction of subplot height <<< independent

    # --- labels ---
    shared_xlabel=r'$\mathbf{x/d_i}$',
    ylabel=r'$y/d_i$',
    cbar_top_text=r'$\mathbf{J_z}$',   # text above colorbar
):
    """
    1x4 panels: each is imshow(Jz) + contour(Ay). Only first has y-label.
    One shared xlabel centered below all panels.
    Colorbar is placed with a custom axis so width & length are controlled independently.
    """

    if len(frames) != 4:
        raise ValueError("frames must have exactly 4 elements.")

    # Diagnostics
    Jz1 = S.Field(0, "Jz")
    Bx1 = S.Field(0, "Bx")
    By1 = S.Field(0, "By")

    time_step = Jz1.getTimesteps()
    time_snap = Jz1.getTimes()

    # Axes
    x_range = np.asarray(Jz1.getAxis("x"), dtype=float)
    y_range = np.asarray(Jz1.getAxis("y"), dtype=float)

    x_min, x_max = x_range.min(), x_range.max()
    y_min, y_max = y_range.min(), y_range.max()
    extent = [x_min/di, x_max/di, y_min/di, y_max/di]

    # Shared norm across all 4 panels
    norm_log = SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=min_val, vmax=max_val)

    # Create 4 panels (no constrained_layout / tight_layout)
    fig, axes = plt.subplots(
        1, 4, figsize=figsize, sharey=True,
        gridspec_kw={"wspace": wspace}
    )

    # Reserve room on right for colorbar: compute right margin from width+pad
    right = 1.0 - (cbar_pad + cbar_width + 0.02)  # extra 0.02 safety margin
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace)

    im_for_cbar = None

    for j, frame in enumerate(frames):
        ax = axes[j]
        ts = time_step[frame]

        # Load fields
        Jz_data = np.squeeze(np.array(Jz1.getData(timestep=ts)), axis=0)
        Bx_data = np.squeeze(np.array(Bx1.getData(timestep=ts)), axis=0)
        By_data = np.squeeze(np.array(By1.getData(timestep=ts)), axis=0)

        # Compute Ay
        Ay = general_compute_Ay(x_range, y_range, Bx_data, By_data)

        # Contour levels
        lo, hi = np.percentile(Ay, contour_percentiles)
        levels = np.linspace(lo, hi, n_contours)

        # Jz_data_yavg = smooth_field_y(Jz_data, y_range, window_phys=0.8, di_unit=di)

        # ---- Plot (ONE imshow only) ----
        im = ax.imshow(
            Jz_data.T,
            origin="lower",
            extent=extent,
            cmap=cmap,
            norm=norm_log,
            aspect=aspect) #, interpolation="nearest")
        # im = ax.imshow(
        #     Jz_data_yavg.T,
        #     origin="lower",
        #     extent=extent,
        #     cmap=cmap,
        #     norm=norm_log,
        #     aspect=aspect) #, interpolation="nearest")

        ax.contour(
            x_range/di, y_range/di, Ay,
            levels=levels,
            colors="k",
            linewidths=1.0,
            linestyles="solid"
        )

        if im_for_cbar is None:
            im_for_cbar = im

        # ---- Per-panel labels ----
        # Remove per-axis x-labels because you want a single global xlabel
        ax.set_xlabel("")

        if j == 0:
            ax.set_ylabel(ylabel, fontsize=20, fontweight="bold")
            ax.tick_params(axis="y", which="both", labelleft=True)
        else:
            ax.tick_params(axis="y", which="both", labelleft=False)

        ax.set_xlim(x_min/di, x_max/di)
        ax.set_ylim(y_min/di, y_max/di)

        # Compute width for annotation
        try:
            _, _, _, _, _, w, _, _, _ = fit_By_tanh_normalized(
                S, frame, time_step, B0, di,
                min_variation=1.0,
                window_size=1500,
                edge_fraction=0.1,
                smooth_sigma=0.0
            )
            w_di = w / di
        except Exception:
            w_di = np.nan

        ax.text(
            0.5, 0.98,
            rf'$\omega_{{pe}}t$={time_snap[frame]:0.1f}, $\lambda/d_i$={w_di:0.1f}',
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=ls-3, color="white",
            bbox=dict(facecolor="black", alpha=0.7)
        )

        panel_labels = ["A", "B", "C", "D"]
        ax.text(
            0.03, 0.91,                       # left + below the 0.98 text
            panel_labels[j],
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=ls+6,
            fontweight="bold",
            color="white",
            bbox=dict(facecolor="black", alpha=0.7) #, pad=0.25)  # same alpha=0.7
        )

        # Your styling helpers (optional)
        try:
            set_spine_linewidth(ax, width=2, spines=['top','bottom','right','left'])
            set_ticks_and_grids(ax,
                major_interval_x=20, major_interval_y=20,
                tick_labelsize=ls+5,
                major_tick_length=10, minor_tick_length=0,
                major_tick_width=5,
                major_grid=False, minor_grid=False
            )
        except NameError:
            pass

    # ---- 1) Shared x-axis label centered under all panels ----
    if hasattr(fig, "supxlabel"):
        fig.supxlabel(shared_xlabel, fontsize=20, fontweight="bold", y=0.02)
    else:
        fig.text(0.5, 0.02, shared_xlabel, ha="center", va="center",
                 fontsize=20, fontweight="bold")

    # ---- 3) Colorbar: control width and length independently via a dedicated axis ----
    # Union bounding box of all 4 axes (in figure coordinates)
    bb = Bbox.union([ax.get_position() for ax in axes])

    # Colorbar axis geometry
    cbar_h = bb.height * float(cbar_length)
    cbar_y0 = bb.y0 + (bb.height - cbar_h) / 2.0
    cbar_x0 = bb.x1 + float(cbar_pad)

    cax = fig.add_axes([cbar_x0, cbar_y0, float(cbar_width), cbar_h])

    cbar_ticks = np.linspace(min_val, max_val, 5)
    cbar = fig.colorbar(
        im_for_cbar,
        cax=cax,
        orientation="vertical",
        ticks=cbar_ticks,
        format=tkr.FormatStrFormatter('%.2f')
    )
    
    cbar.ax.tick_params(labelsize=ls+1, width=4.5, length=8)
    cbar.ax.tick_params(which='both', direction='in', right=True)

    # --- add border like your plots ---
    set_spine_linewidth(cbar.ax, width=2, spines=['top', 'bottom', 'left', 'right'])

    # --- (optional but recommended) ensure the colorbar outline matches too ---
    cbar.outline.set_linewidth(2)
    
    # ---- 2) Put j_z on top of the colorbar (in the empty space above it) ----
    cbar.ax.text(
        0.8, 1.01, cbar_top_text,
        transform=cbar.ax.transAxes,
        ha="center", va="bottom",
        fontsize=22, fontweight="bold", color="k"
    )

    return fig, axes, cbar

############################################################################################
### y-window average

from scipy.ndimage import uniform_filter1d
import numpy as np

def smooth_field_y(field_data, y_axis_coords, window_phys=0.8, di_unit=1.0):
    """
    Performs a window-average (smoothing) along the y-axis of a 2D field.
    
    Parameters:
    -----------
    field_data : np.ndarray
        The 2D array to be smoothed (e.g., Jz_data).
    y_axis_coords : np.ndarray
        The 1D array representing the y-axis physical coordinates.
    window_phys : float
        The physical size of the smoothing window (default: 0.8).
    di_unit : float
        The normalization factor (di) to scale the window (default: 1.0).
        
    Returns:
    --------
    smoothed_data : np.ndarray
        The field smoothed along the identified y-axis.
    """
    
    # 1. Calculate window size in grid points
    window_size_phys = window_phys * di_unit
    dy = float(np.nanmedian(np.diff(y_axis_coords)))
    
    if dy <= 0:
        raise ValueError("y_axis_coords must be strictly increasing.")

    nwin = int(np.round(window_size_phys / dy))
    nwin = max(1, nwin)
    
    # Force odd window for symmetry
    if (nwin % 2) == 0:
        nwin += 1

    # 2. Automatically identify the y-axis index
    ny = len(y_axis_coords)
    if field_data.shape[1] == ny:
        target_axis = 1
    elif field_data.shape[0] == ny:
        target_axis = 0
    else:
        raise ValueError(
            f"Cannot identify y-axis. Field shape {field_data.shape} "
            f"does not match coordinate length {ny}."
        )

    # 3. Apply the uniform filter
    smoothed_data = uniform_filter1d(field_data, size=nwin, axis=target_axis, mode="nearest")
    
    return smoothed_data