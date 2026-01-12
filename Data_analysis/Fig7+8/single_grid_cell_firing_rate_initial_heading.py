"""
Script to plot grid cell firing rates sorted by initial heading.

This script creates trial matrices showing firing rate of grid cells as a function
of angular direction relative to the lever during search (left), at lever (middle),
and homing (right) phases. Trials are sorted by the initial heading of the mouse as
it leaves the home base at the start of the trial.

Key differences from decoded error analysis:
- Uses actual grid cell firing rates (Hz) from pickle files
- Angular direction relative to lever (from lever-centered reference frame)
- Selects top 3 grid cells per session by mean vector length
- Red ticks show peak firing direction for each trial
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import pickle
import math
import collections
from scipy import stats
from scipy.stats import pearsonr, gaussian_kde
from tqdm import tqdm
import sys
import os
from astropy.stats import circcorrcoef
from sklearn.linear_model import LinearRegression
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Import project setup
exec(open('E:/GitHub/Peng_et.al_2025_noInt/setup_project.py').read())
exec(open('E:/GitHub/Peng_et.al_2025_noInt/generic_plot_functions.py').read())

# ============================================================================
# CONFIGURATION
# ============================================================================

GLOBALFONTSIZE = 12
GLOBALCONV = True  # Apply Gaussian smoothing
GLOBALSPEEDFILTER = True  # Filter by speed > 10 cm/s

# Color scheme
colors = ['#1a2a6c', '#b21f1f', '#fdbb2d']
modelCmap = LinearSegmentedColormap.from_list("custom_colormap", colors, N=256)

# Bin configuration (must match pickle file structure)
DEGPERBIN = 10
NUM_BINS = 36
BIN_EDGES = np.arange(-np.pi, np.pi + DEGPERBIN/360*2*np.pi, DEGPERBIN/360*2*np.pi)
GLOBAL_BINS = np.linspace(-np.pi, np.pi - (2*np.pi/36), 36)

# ============================================================================
# SETUP AND DATA LOADING
# ============================================================================

print("Setting up project...")
projectName, dataPath, dlcModelPath, myProject, sSessions = setup_project_session_lists(
    projectName="autopi_mec",
    dataPath=PROJECT_DATA_PATH,
    dlcModelPath=""
)

# Load behavioral data
print("Loading behavioral data...")
fn = myProject.dataPath + '/results/behavior_180_EastReferenceQuadrant.csv'
res = pd.read_csv(fn)
res = res[res.valid]

# Load grid cell data
print("Loading grid cell data...")
fn = myProject.dataPath + "/results/cells.csv"
gc = pd.read_csv(fn)
gc = gc.loc[gc["gridCell_AND"], :]

# Load session data for initial heading calculation
print("Loading session data...")
fn = myProject.dataPath + '/results/allSessionDf_with_leverVector_and_last_cohort.csv'
allSessionDf = pd.read_csv(fn, index_col=0)
allSessionDf['light'] = allSessionDf['condition'].apply(lambda x: x.split('_')[1])
allSessionDf['cond_noLight'] = allSessionDf['condition'].apply(lambda x: x.split('_')[0])

full_all_sessions = allSessionDf.copy()

# ============================================================================
# NESTED DICT DEFINITION (Required for unpickling)
# ============================================================================

def nested_dict():
    """Recursive defaultdict for nested dictionary structures in pickle files."""
    return collections.defaultdict(nested_dict)


# ============================================================================
# LOAD FIRING RATE PICKLE FILES
# ============================================================================

print("Loading firing rate pickle files...")

# Trial-by-trial firing rate histograms
fn = myProject.dataPath + "/results/myLeverCellStatsTrialMatrix_all.pickle"
try:
    with open(fn, 'rb') as fp:
        hdLeverCenteredTrialMatrix = pickle.load(fp)
    print(f"  Loaded trial matrix pickle: {len(hdLeverCenteredTrialMatrix)} cells")
except FileNotFoundError:
    print(f"  ERROR: Could not find {fn}")
    print("  This pickle file is required for firing rate analysis.")
    sys.exit(1)

# Mean statistics for cell selection
fn = myProject.dataPath + "/results/myLeverCellStatsSplitByMedian_homingDir_all.pickle"
try:
    with open(fn, 'rb') as fp:
        hdLeverCenteredLeftRightHeadingError = pickle.load(fp)
    print(f"  Loaded statistics pickle: {len(hdLeverCenteredLeftRightHeadingError)} cells")
except FileNotFoundError:
    print(f"  ERROR: Could not find {fn}")
    print("  This pickle file is required for cell selection.")
    sys.exit(1)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def centerAngles(a):
    """Center angles on their circular mean (or just return as-is for no centering)."""
    # For consistency with original script, just return angles as-is
    return a


def interpolate_histogram(hist):
    """Interpolate NaN values in histogram using neighboring values."""
    hist = np.array(hist, dtype=float).copy()  # Make defensive copy
    nan_indices = np.isnan(hist)
    if not np.any(nan_indices):
        return hist

    indices = np.arange(len(hist))
    non_nan_indices = ~nan_indices

    if not np.any(non_nan_indices):
        # All NaN, return zeros
        return np.zeros_like(hist)

    hist[nan_indices] = np.interp(indices[nan_indices],
                                   indices[non_nan_indices],
                                   hist[non_nan_indices])
    return hist


def smooth_heatmap(heatmap, sigma=1, ksize=6):
    """Smooth heatmap using Gaussian kernel with circular boundary conditions."""
    from scipy.ndimage import convolve1d
    from scipy.signal.windows import gaussian

    size = int(ksize * sigma)
    gaussian_kernel = gaussian(2 * size + 1, sigma)
    gaussian_kernel /= gaussian_kernel.sum()

    if heatmap.ndim == 1:
        heatmap = np.atleast_2d(heatmap)

    smoothed_heatmap = convolve1d(heatmap, gaussian_kernel, mode='wrap', axis=1)
    return smoothed_heatmap.squeeze()


def normalize_2d_array(arr):
    """Normalize values in each row to range [0, 1]."""
    np_arr = np.array(arr, dtype=float)
    min_vals = np.min(np_arr, axis=1, keepdims=True)
    max_vals = np.max(np_arr, axis=1, keepdims=True)
    max_vals[max_vals == min_vals] = 1  # Avoid division by zero
    normalized_arr = (np_arr - min_vals) / (max_vals - min_vals)
    return normalized_arr


# ============================================================================
# INITIAL HEADING CALCULATION
# ============================================================================

def calculate_initial_heading(sessionSlice, time_window=0.5):
    """
    Calculate initial heading for each trial as average of first few seconds of search.

    Parameters:
    -----------
    sessionSlice : DataFrame
        Session data containing trials
    time_window : float
        Time window in seconds to average over (default: 1 second)

    Returns:
    --------
    DataFrame with initialHeading column added
    """
    print(f"  Calculating initial heading (averaging first {time_window} seconds of search)...")

    initial_headings = []
    trial_numbers = []

    # Process each trial
    for trial in sessionSlice['trialNo'].unique():
        # Get search phase data for this trial
        trial_data = sessionSlice[
            (sessionSlice['trialNo'] == trial) &
            (sessionSlice['condition'] == 'searchToLeverPath_dark')
        ].copy()

        if len(trial_data) == 0:
            continue

        # Sort by time within path
        trial_data = trial_data.sort_values('withinPathTime')

        # Get data from first few seconds
        initial_data = trial_data[trial_data['withinPathTime'] <= time_window]

        if len(initial_data) > 0:
            # Calculate heading from movement direction
            if 'xPose' in initial_data.columns and 'yPose' in initial_data.columns:
                # Calculate from position changes
                dx = np.diff(initial_data['xPose'].values)
                dy = np.diff(initial_data['yPose'].values)
                headings = np.arctan2(dy, dx)
            elif 'hdPose' in initial_data.columns:
                # Fall back to hdPose if position data not available
                headings = initial_data['hdPose'].values
            else:
                continue

            # Remove NaN and zero values
            headings = headings[~np.isnan(headings)]
            headings = headings[headings != 0]

            if len(headings) > 0:
                # Calculate circular mean
                sin_sum = np.sum(np.sin(headings))
                cos_sum = np.sum(np.cos(headings))
                initial_heading = np.arctan2(sin_sum, cos_sum)
                initial_heading = math.remainder(initial_heading, math.tau)
                initial_heading = centerAngles(np.array([initial_heading]))[0]

                initial_headings.append(initial_heading)
                trial_numbers.append(trial)

    # Create DataFrame with results
    initial_heading_df = pd.DataFrame({
        'trialNo': trial_numbers,
        'initialHeading': initial_headings
    })

    if len(initial_headings) > 0:
        non_zero_count = np.sum(np.abs(initial_headings) > 0.01)
        print(f"    Found {len(initial_headings)} trials with initial heading data")
        print(f"    Non-zero headings: {non_zero_count}/{len(initial_headings)}")
        print(f"    Initial heading range: [{np.min(initial_headings):.3f}, {np.max(initial_headings):.3f}] radians")
    else:
        print("    WARNING: No initial heading data found!")

    return initial_heading_df


# ============================================================================
# CELL SELECTION FUNCTIONS
# ============================================================================

def get_mean_vector_length_from_pickle(cellID, pickle_dict):
    """
    Extract mean vector length for a cell from the pickle file.

    Parameters:
    -----------
    cellID : str
        Cell cluster ID (e.g., 'jp486-24032023-0108_608')
    pickle_dict : dict
        hdLeverCenteredLeftRightHeadingError dictionary

    Returns:
    --------
    float : Mean vector length, or NaN if not found
    """
    try:
        # Try to get from 'dark', 'all' condition, either median split
        hd_stats = pickle_dict[cellID]['dark']['all']['belowMedianIntervals']['hd_score']
        # hd_score is tuple: (mean_dir_rad, mean_dir_deg, mean_vector_length, peak_rad, peak_rate)
        mean_vector_length = hd_stats[2]
        return mean_vector_length
    except (KeyError, IndexError, TypeError):
        return np.nan


def select_top_cells_by_vector_length(session_name, pickle_dict, grid_cells_df, n=3):
    """
    Select top N grid cells for a session based on mean vector length.

    Parameters:
    -----------
    session_name : str
        Session name (e.g., 'jp486-24032023-0108')
    pickle_dict : dict
        hdLeverCenteredLeftRightHeadingError dictionary
    grid_cells_df : DataFrame
        Grid cell metadata
    n : int
        Number of cells to select (default: 3)

    Returns:
    --------
    list : Cell IDs sorted by vector length (best first)
    """
    # Get all grid cells for this session
    session_cells = grid_cells_df[grid_cells_df['session'] == session_name].copy()

    if len(session_cells) == 0:
        print(f"  WARNING: No grid cells found for session {session_name}")
        return []

    # Extract vector lengths
    vector_lengths = []
    cell_ids = []

    for idx, row in session_cells.iterrows():
        # Get cell ID (already in correct format: session_clusterNumber)
        cell_id = row['cluId']

        # Get vector length
        mvl = get_mean_vector_length_from_pickle(cell_id, pickle_dict)

        if not np.isnan(mvl):
            vector_lengths.append(mvl)
            cell_ids.append(cell_id)

    if len(cell_ids) == 0:
        print(f"  WARNING: No cells with vector length data for session {session_name}")
        return []

    # Sort by vector length (descending)
    sorted_indices = np.argsort(vector_lengths)[::-1]
    top_cells = [cell_ids[i] for i in sorted_indices[:n]]
    top_mvls = [vector_lengths[i] for i in sorted_indices[:n]]

    print(f"  Selected top {len(top_cells)} cells for {session_name}:")
    for i, (cell, mvl) in enumerate(zip(top_cells, top_mvls)):
        print(f"    {i+1}. {cell} (MVL: {mvl:.3f})")

    return top_cells


# ============================================================================
# FIRING RATE HISTOGRAM EXTRACTION
# ============================================================================

def extract_firing_rate_trial_matrix(cellID, condition, trial_matrix_dict, res_df,
                                      initial_heading_df, light='dark', interval='all'):
    """
    Extract firing rate histograms for all trials of a cell in a specific condition.

    Parameters:
    -----------
    cellID : str
        Cell cluster ID
    condition : str
        Condition name (e.g., 'searchToLeverPath_dark')
    trial_matrix_dict : dict
        hdLeverCenteredTrialMatrix dictionary
    res_df : DataFrame
        Behavioral results dataframe
    initial_heading_df : DataFrame
        Initial heading dataframe with trialNo and initialHeading columns
    light : str
        Light condition ('dark' or 'light')
    interval : str
        Interval type ('all')

    Returns:
    --------
    DataFrame with columns: trialNo, initialHeading, bin_1, ..., bin_36
    """
    # Get session name from cell ID
    session_name = '_'.join(cellID.split('_')[:-1])

    # Get all valid trials for this session and condition
    session_trials = res_df[
        (res_df['sessionName'] == session_name) &
        (res_df['valid'] == True)
    ]['trialNo'].unique()

    histos_holder = []
    trial_holder = []

    for trial_no in session_trials:
        try:
            # Extract firing rate histogram for this trial
            trial_no_key = int(trial_no)  # Ensure Python int, not numpy.int64
            trial_data = trial_matrix_dict[cellID][light][interval][trial_no_key]
            histo = trial_data['Hist']

            # Check if histogram is valid (not all NaN, not all zeros)
            if histo is not None and len(histo) == NUM_BINS:
                if not (np.all(np.isnan(histo)) or np.all(histo == 0)):
                    histos_holder.append(histo)
                    trial_holder.append(trial_no)
        except (KeyError, TypeError):
            # Trial not found in pickle file, skip
            continue

    if len(histos_holder) == 0:
        return pd.DataFrame()  # No valid trials

    # Create DataFrame
    histo_df = pd.DataFrame(histos_holder, columns=[f'bin_{i}' for i in range(1, NUM_BINS+1)])
    trial_df = pd.DataFrame({'trialNo': trial_holder})

    combined_df = pd.concat([trial_df, histo_df], axis=1)

    # Merge with initial heading
    combined_df = pd.merge(combined_df, initial_heading_df, on='trialNo', how='left')

    # Filter out trials without initial heading
    combined_df = combined_df.dropna(subset=['initialHeading'])

    return combined_df


def get_peak_firing_from_slice(inputDf, convolution=True, gaus=True):
    """
    Get peak firing direction for each trial from firing rate histograms.

    Parameters:
    -----------
    inputDf : DataFrame
        DataFrame with last 36 columns as firing rate histograms
    convolution : bool
        Whether to apply smoothing before finding peak
    gaus : bool
        Whether to use Gaussian smoothing

    Returns:
    --------
    array : Peak firing directions in radians for each trial
    """
    if len(inputDf) == 0:
        return np.array([])

    # Sort by initial heading
    sortSlice = inputDf.sort_values(by='initialHeading', ascending=True)

    # Find global peak position for alignment
    heatmap_slice = np.array(sortSlice.iloc[:, -NUM_BINS:])
    shiftIndex = np.argmax(np.nanmean(heatmap_slice, axis=0))

    peak_indices = []

    for i in range(len(sortSlice)):
        # Extract firing rate histogram for this trial
        histo = np.array(sortSlice.iloc[i, -NUM_BINS:]).astype(float)

        # Interpolate NaN values
        histo = interpolate_histogram(histo)

        # Align to global peak
        histo = np.roll(histo, 18 - shiftIndex)

        # Apply smoothing if requested
        if convolution and gaus:
            histo = smooth_heatmap(histo)

        # Find peak
        peak_idx = np.argmax(histo)
        peak_indices.append(peak_idx)

    # Convert peak indices to radians
    # After alignment, bin 18 corresponds to angle 0
    # Each bin is 10° = π/18 radians
    peak_indices = np.array(peak_indices)
    peak_shifts = peak_indices - 18  # Shift relative to center
    peak_radians = peak_shifts * (DEGPERBIN * np.pi / 180)

    # Wrap to [-π, π]
    peak_radians = np.array([math.remainder(x, math.tau) for x in peak_radians])
    peak_radians = centerAngles(peak_radians)

    return peak_radians


# ============================================================================
# CIRCULAR CORRELATION FUNCTIONS
# ============================================================================

def homing_angle_corr_stats(x, y, signLevel=0.025):
    """
    Calculate circular correlation statistics between two angular variables.

    Parameters:
    -----------
    x : array-like
        First angular variable (radians)
    y : array-like
        Second angular variable (radians)
    signLevel : float
        Significance level for permutation test (default: 0.025)

    Returns:
    --------
    tuple : (realR, slope, meanShuffleR, significant, pValue)
    """
    realR = circcorrcoef(x, y)
    anglesCentered = centerAngles(x)
    model = LinearRegression()
    model.fit(anglesCentered[:, np.newaxis], y)
    slope = model.coef_[0]

    xr = x.copy()
    n = 10000
    shufR = np.zeros(n)

    for i in range(n):
        np.random.shuffle(xr)
        shufR[i] = circcorrcoef(xr, y)

    # Calculate two-tailed p-value
    pValue = np.sum(np.abs(shufR) >= np.abs(realR)) / n

    signPlus = np.quantile(shufR, 1 - signLevel)
    signMinus = np.quantile(shufR, signLevel)

    if realR < signMinus or realR > signPlus:
        return (realR, slope, np.nanmean(shufR), True, pValue)
    else:
        return (realR, slope, np.nanmean(shufR), False, pValue)


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plotRegressionLine(ax, x, y, regColor='#ff9e00'):
    """Plot regression line for circular correlation."""
    anglesCentered = centerAngles(x)
    model = LinearRegression()
    model.fit(anglesCentered[:, np.newaxis], y)

    x_line = np.array(np.linspace(x.min() - 1, x.max() + 1, 100))
    y_line = model.predict(x_line[:, np.newaxis])

    ax.plot(x_line, y_line, color=regColor, lw=2.5, linestyle='-')


def plot_kdeplot(ax, inputDf, c='#12c2e9', ylabel='Peak firing dir.',
                 xlabel='Initial heading', set_ylabel=True, signLevel=0.025):
    """
    Plot KDE plot with scatter and regression line for circular correlation.

    Parameters:
    -----------
    ax : matplotlib axis
    inputDf : DataFrame
        Must have 'initialHeading' and 'peakFiringDir' columns
    c : str
        Color for plot
    ylabel : str
        Y-axis label
    xlabel : str
        X-axis label
    set_ylabel : bool
        Whether to set y-axis label
    signLevel : float
        Significance level
    """
    xVal = inputDf['initialHeading'].values
    yVal = inputDf['peakFiringDir'].values

    # Create 2D KDE plot
    try:
        xy = np.vstack([xVal, yVal])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x_sorted, y_sorted, z_sorted = xVal[idx], yVal[idx], z[idx]
        ax.scatter(x_sorted, y_sorted, c=z_sorted, s=40, cmap='Blues',
                  edgecolor='black', linewidth=1, alpha=0.8)
    except:
        ax.scatter(xVal, yVal, s=40, edgecolor='black', linewidth=1,
                  color=c, alpha=0.8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plotRegressionLine(ax, x=xVal, y=yVal, regColor='#ff9e00')

    if set_ylabel:
        ax.set_yticks(ticks=[-np.pi, 0, np.pi])
        ax.set_yticklabels(["-$\\pi$", "0", "$\\pi$"], fontsize=GLOBALFONTSIZE)
        ax.set_ylabel(ylabel, fontsize=GLOBALFONTSIZE, labelpad=1)
    else:
        ax.set_yticks([])
        ax.set_yticklabels([], fontsize=GLOBALFONTSIZE)
        ax.set_ylabel('', fontsize=GLOBALFONTSIZE)

    ax.set_xticks(ticks=[-np.pi, 0, np.pi])
    ax.set_xticklabels(["-$\\pi$", "0", "$\\pi$"], fontsize=GLOBALFONTSIZE)
    ax.set_xlabel(xlabel, fontsize=GLOBALFONTSIZE)
    ax.tick_params(axis='both', which='both', labelsize=GLOBALFONTSIZE)

    realR, slope, sigN, _, pValue = homing_angle_corr_stats(xVal, yVal, signLevel=signLevel)

    if sigN:
        label_x = 0.65
        label_y1 = 0.05
        label_y2 = 0.17
        label_y3 = 0.29
        ax.text(label_x, label_y1, 's: ' + str(round(slope, 3)), fontsize=GLOBALFONTSIZE, transform=ax.transAxes)
        ax.text(label_x, label_y2, f'r: ' + str(round(realR, 3)), fontsize=GLOBALFONTSIZE, transform=ax.transAxes)
        ax.text(label_x, label_y3, f'p: ' + str(round(pValue, 4)), fontsize=GLOBALFONTSIZE, transform=ax.transAxes)
    else:
        ax.text(0.1, 0.8, 'Not sig.', fontsize=GLOBALFONTSIZE, transform=ax.transAxes)

    borders = np.pi
    ax.set_xlim(-borders, borders)
    ax.set_ylim(-borders, borders)


def plot_correlation_distribution(ax, accumulated_stats_df, xlabel='Peak-initial heading corr. (r)',
                                   ylabel='Cells', legend=True, ylim=50,
                                   title='', set_ylabel=True):
    """Plot histogram of circular correlation coefficients accumulated across cells."""
    if len(accumulated_stats_df) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # Map boolean to string labels
    accumulated_stats_df['sig_label'] = accumulated_stats_df['sig'].map({True: 'Significant', False: 'Non sign.'})

    # Color palette
    palette = {'Significant': '#1a659e', 'Non sign.': '#ff6b35'}
    order = ['Significant', 'Non sign.']

    # Create histogram
    sns.histplot(
        data=accumulated_stats_df, x='realR', hue='sig_label',
        bins=10, kde=False, palette=palette,
        legend=legend, hue_order=order, ax=ax
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if title:
        ax.set_title(title, fontsize=GLOBALFONTSIZE, fontweight='bold', y=1.0)

    if set_ylabel:
        ax.set_ylabel(ylabel, fontsize=GLOBALFONTSIZE, labelpad=1)
    else:
        ax.set_ylabel('', fontsize=GLOBALFONTSIZE)
        ax.set_yticks([])

    ax.set_xlabel(xlabel, fontsize=GLOBALFONTSIZE)
    ax.set_xticks(ticks=[-1, 0, 1])
    ax.set_xticklabels(['-1', '0', '1'], fontsize=GLOBALFONTSIZE)
    ax.set_ylim(0, ylim)

    # Add vertical line at zero
    ax.axvline(x=0, color='#8A817C', linestyle='--', lw=2, ymax=0.68)

    # Statistical test (Wilcoxon signed-rank test against 0)
    realR_values = accumulated_stats_df['realR'].values
    if len(realR_values) > 0:
        stat, p_value = stats.wilcoxon(realR_values)

        n_cells = len(accumulated_stats_df)
        ax.text(0.07, 0.9, f'N = {n_cells}', fontsize=GLOBALFONTSIZE, transform=ax.transAxes)

        if p_value < 0.0001:
            p_text = 'p < 0.0001'
        else:
            p_text = f'p = {round(p_value, 4)}'
        ax.text(0.07, 0.78, p_text, fontsize=GLOBALFONTSIZE, transform=ax.transAxes, fontweight='bold')

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles, labels, fontsize=GLOBALFONTSIZE - 1,
            frameon=False, loc='lower left',
            bbox_to_anchor=(-0.01, 0.65)
        )

    ax.tick_params(axis='both', which='both', labelsize=GLOBALFONTSIZE)


def plot_firing_rate_heatmap(ax, cellID, condition, trial_matrix_dict, res_df,
                             initial_heading_df, set_ticks=False, ylabel='', xlabel=''):
    """
    Plot firing rate heatmap for a single cell and condition, sorted by initial heading.

    Parameters:
    -----------
    ax : matplotlib axis
    cellID : str
        Cell cluster ID
    condition : str
        Condition name (without _light/_dark suffix)
    trial_matrix_dict : dict
        hdLeverCenteredTrialMatrix dictionary
    res_df : DataFrame
        Behavioral results
    initial_heading_df : DataFrame
        Initial heading per trial
    set_ticks : bool
        Whether to show peak markers
    ylabel : str
        Y-axis label
    xlabel : str
        X-axis label
    """
    # Extract trial matrix for this condition
    condition_full = f'{condition}_dark'
    trial_df = extract_firing_rate_trial_matrix(
        cellID, condition_full, trial_matrix_dict, res_df, initial_heading_df, light='dark'
    )

    if len(trial_df) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # Sort by initial heading
    trial_df = trial_df.sort_values(by='initialHeading', ascending=True)

    # Extract histogram matrix
    heatmap_slice = np.array(trial_df.iloc[:, -NUM_BINS:])

    # Interpolate NaN values
    for i in range(len(heatmap_slice)):
        heatmap_slice[i] = interpolate_histogram(heatmap_slice[i])

    # Find global peak for alignment
    shiftIndex = np.argmax(np.nanmean(heatmap_slice, axis=0))

    # Align all histograms to global peak
    heatmap_slice = np.roll(heatmap_slice, 18 - shiftIndex, axis=1)

    # Apply Gaussian smoothing if requested
    if GLOBALCONV:
        heatmap_slice = smooth_heatmap(heatmap_slice)

    # Normalize each trial
    heatmap_slice = normalize_2d_array(heatmap_slice)

    # Get peak indices for markers
    maxInd = np.argmax(heatmap_slice, axis=1)

    # Plot heatmap
    sns.heatmap(heatmap_slice, cmap=modelCmap, cbar=False, ax=ax)

    # Add peak markers
    if set_ticks:
        for i, max_index in enumerate(maxInd):
            ax.plot(max_index + 0.5, i + 0.5, marker='^', color='red', ms=3)

    # Format axes
    ax.set_xticks(ticks=[0, 18, 36])
    ax.set_xticklabels(["-$\\pi$", "0", "$\\pi$"], fontsize=GLOBALFONTSIZE)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylabel(ylabel, fontsize=GLOBALFONTSIZE)
    ax.set_xlabel(xlabel, fontsize=GLOBALFONTSIZE)


# ============================================================================
# AGGREGATE STATISTICS ACROSS SESSIONS
# ============================================================================

def compute_aggregate_statistics_all_sessions(session_list,
                                              condition_to_plot,
                                              pickle_dict_trial_matrix,
                                              pickle_dict_stats,
                                              gc_df,
                                              res_df,
                                              full_sessions_df,
                                              max_cells_per_session=None):
    """
    Compute correlation statistics for all grid cells across multiple sessions.

    Parameters:
    -----------
    session_list : list of str
        List of session names to process (e.g., useAble from setup_project.py)
    condition_to_plot : str
        Behavioral phase to analyze ('searchToLeverPath', 'atLever', etc.)
    pickle_dict_trial_matrix : dict
        hdLeverCenteredTrialMatrix dictionary
    pickle_dict_stats : dict
        hdLeverCenteredLeftRightHeadingError dictionary
    gc_df : DataFrame
        Grid cell metadata (cells.csv)
    res_df : DataFrame
        Behavioral results dataframe
    full_sessions_df : DataFrame
        Full session data for initial heading calculation
    max_cells_per_session : int, optional
        Maximum cells to include per session (None = all grid cells)

    Returns:
    --------
    DataFrame with columns: sessionName, cellID, cell_idx_in_session, realR, slope, sig, pValue
    """
    all_stats = []

    print(f"\n{'='*80}")
    print(f"Computing aggregate statistics across {len(session_list)} sessions")
    print(f"Condition: {condition_to_plot}")
    if max_cells_per_session:
        print(f"Max cells per session: {max_cells_per_session}")
    else:
        print(f"Including all grid cells per session")
    print(f"{'='*80}\n")

    for session_idx, session_name in enumerate(session_list):
        print(f"[{session_idx+1}/{len(session_list)}] Processing {session_name}...")

        try:
            # Select grid cells for this session
            session_cells = select_top_cells_by_vector_length(
                session_name,
                pickle_dict_stats,
                gc_df,
                n=max_cells_per_session if max_cells_per_session else 1000  # Large number to get all
            )

            if len(session_cells) == 0:
                print(f"  ⚠ No grid cells found, skipping")
                continue

            # Limit if requested
            if max_cells_per_session:
                session_cells = session_cells[:min(max_cells_per_session, len(session_cells))]

            print(f"  Found {len(session_cells)} grid cells")

            # Calculate initial heading for this session
            session_slice = full_sessions_df[
                full_sessions_df.session == session_name
            ].reset_index()

            if GLOBALSPEEDFILTER:
                session_slice = session_slice[session_slice.speed > 10]

            initial_heading_df = calculate_initial_heading(session_slice)

            if len(initial_heading_df) == 0:
                print(f"  ⚠ No initial heading data, skipping")
                continue

            print(f"  Calculated initial heading for {len(initial_heading_df)} trials")

            # Process each cell
            cells_processed = 0
            for cell_idx, cell_id in enumerate(session_cells):
                try:
                    condition_full = f'{condition_to_plot}_dark'
                    trial_df = extract_firing_rate_trial_matrix(
                        cell_id, condition_full,
                        pickle_dict_trial_matrix, res_df, initial_heading_df
                    )

                    if len(trial_df) == 0:
                        continue

                    peak_dirs = get_peak_firing_from_slice(
                        trial_df, convolution=GLOBALCONV
                    )

                    if len(peak_dirs) == 0:
                        continue

                    trial_df_sorted = trial_df.sort_values(
                        by='initialHeading', ascending=True
                    )
                    trial_df_sorted['peakFiringDir'] = peak_dirs

                    xVal = trial_df_sorted['initialHeading'].values
                    yVal = trial_df_sorted['peakFiringDir'].values
                    realR, slope, _, sig, pValue = homing_angle_corr_stats(xVal, yVal)

                    all_stats.append({
                        'sessionName': session_name,
                        'cellID': cell_id,
                        'cell_idx_in_session': cell_idx,
                        'realR': realR,
                        'slope': slope,
                        'sig': sig,
                        'pValue': pValue
                    })

                    cells_processed += 1

                except Exception as e:
                    # Silently skip cells that fail
                    continue

            print(f"  ✓ Successfully processed {cells_processed}/{len(session_cells)} cells")

        except Exception as e:
            print(f"  ✗ Error processing session: {e}")
            continue

    if len(all_stats) == 0:
        print("\n⚠ WARNING: No statistics computed across any session!")
        return pd.DataFrame()

    stats_df = pd.DataFrame(all_stats)

    print(f"\n{'='*80}")
    print(f"Aggregate Statistics Summary:")
    print(f"  Total sessions processed: {stats_df['sessionName'].nunique()}")
    print(f"  Total cells analyzed: {len(stats_df)}")
    print(f"  Significant correlations: {stats_df['sig'].sum()} ({100*stats_df['sig'].mean():.1f}%)")
    print(f"  Mean correlation: {stats_df['realR'].mean():.3f} ± {stats_df['realR'].std():.3f}")
    print(f"{'='*80}\n")

    return stats_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def create_multi_cell_plot(session_name,
                           condition_to_plot='searchToLeverPath',
                           n_cells_total=10,
                           cells_to_display=[0, 1, 2],
                           aggregate_across_sessions=True,
                           max_cells_per_session=None,
                           output_path=None,
                           height_per_row=4):
    """
    Create 3×3 grid plot showing firing rates for selected grid cells.

    Layout:
    -------
    Row 0: [Cell 1 Heatmap] [Cell 2 Heatmap] [Cell 3 Heatmap]
    Row 1: [Cell 1 Corr.  ] [Cell 2 Corr.  ] [Cell 3 Corr.  ]
    Row 2: [    Aggregate Histogram (spans all 3 columns)    ]

    Parameters:
    -----------
    session_name : str
        Session name to analyze (used for Rows 0-1 display)
    condition_to_plot : str
        Behavioral phase to visualize (default: 'searchToLeverPath')
        Options: 'searchToLeverPath', 'atLever', 'homingFromLeavingLeverToPeriphery'
    n_cells_total : int
        Number of top MVL cells from session_name to select from (default: 10)
        Used only for selecting cells_to_display
    cells_to_display : list of int
        Indices of 3 cells (from top N) to display in rows 0-1 (default: [0, 1, 2])
        Example: [0, 1, 2] = top 3 cells, [0, 5, 9] = 1st, 6th, 10th cells
    aggregate_across_sessions : bool
        If True, Row 2 aggregates across all useAble sessions (default: True)
        If False, Row 2 shows only top N cells from session_name
    max_cells_per_session : int, optional
        When aggregate_across_sessions=True, limit cells per session (default: None = all)
        Example: 20 = include top 20 MVL cells from each session
    output_path : str, optional
        Path to save figure
    height_per_row : float
        Height in inches per row (default: 4)
    """
    print(f"\n{'='*80}")
    print(f"Processing session: {session_name}")
    print(f"  Condition: {condition_to_plot}")
    print(f"  Total cells for statistics: {n_cells_total}")
    print(f"  Cells to display: {cells_to_display}")
    print(f"{'='*80}")

    # Validate cells_to_display parameter
    if len(cells_to_display) != 3:
        raise ValueError(f"cells_to_display must have exactly 3 elements, got {len(cells_to_display)}")

    if any(idx < 0 or idx >= n_cells_total for idx in cells_to_display):
        raise ValueError(f"cells_to_display indices must be in range [0, {n_cells_total-1}]")

    # Select top N cells by vector length
    top_cells = select_top_cells_by_vector_length(
        session_name, hdLeverCenteredLeftRightHeadingError, gc, n=n_cells_total
    )

    if len(top_cells) == 0:
        print(f"  ERROR: No valid cells found for session {session_name}")
        return None

    if len(top_cells) < n_cells_total:
        print(f"  WARNING: Only {len(top_cells)} cells found, requested {n_cells_total}")
        n_cells_total = len(top_cells)

    # Validate that selected cells exist
    if max(cells_to_display) >= len(top_cells):
        raise ValueError(f"Cell index {max(cells_to_display)} out of range, only {len(top_cells)} cells available")

    # Calculate initial heading for this session
    print(f"\nCalculating initial heading for {session_name}...")
    session_slice = full_all_sessions[full_all_sessions.session == session_name].reset_index()
    if GLOBALSPEEDFILTER:
        session_slice = session_slice[session_slice.speed > 10]
    initial_heading_df = calculate_initial_heading(session_slice)

    if len(initial_heading_df) == 0:
        print(f"  ERROR: No initial heading data for session {session_name}")
        return None

    # Create figure with 3×3 grid
    fig_width = 12
    fig_height = 3 * height_per_row
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(3, 3, figure=fig,
                          wspace=0.3,
                          hspace=0.4,
                          height_ratios=[1, 1, 1])

    # Determine condition label for title
    condition_label = condition_to_plot.replace('ToLeverPath', '').replace('FromLeavingLeverToPeriphery', '').replace('atLever', 'At Lever').capitalize()

    # ===== ROW 0: TRIAL MATRIX HEATMAPS FOR 3 SELECTED CELLS =====

    print(f"\nRow 0: Plotting trial matrices...")
    for col_idx, display_idx in enumerate(cells_to_display):
        cell_id = top_cells[display_idx]
        print(f"  Column {col_idx}: Cell {display_idx+1}/{n_cells_total} ({cell_id})")

        ax = fig.add_subplot(gs[0, col_idx])

        ylabel_text = f'Cell {display_idx+1}\nTrials' if col_idx == 0 else f'Cell {display_idx+1}'

        plot_firing_rate_heatmap(
            ax, cell_id, condition_to_plot, hdLeverCenteredTrialMatrix,
            res, initial_heading_df,
            set_ticks=True,
            ylabel=ylabel_text,
            xlabel=''
        )

        if col_idx == 0:
            ax.set_title(condition_label, fontsize=GLOBALFONTSIZE + 2, loc='left')

    # ===== ROW 1: CORRELATION PLOTS FOR 3 SELECTED CELLS =====

    print(f"\nRow 1: Plotting correlations...")
    displayed_cell_stats = []

    for col_idx, display_idx in enumerate(cells_to_display):
        cell_id = top_cells[display_idx]
        ax = fig.add_subplot(gs[1, col_idx])

        try:
            # Extract trial data for this cell and condition
            condition_full = f'{condition_to_plot}_dark'
            trial_df = extract_firing_rate_trial_matrix(
                cell_id, condition_full,
                hdLeverCenteredTrialMatrix, res, initial_heading_df
            )

            if len(trial_df) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # Calculate peak firing directions
            peak_dirs = get_peak_firing_from_slice(trial_df, convolution=GLOBALCONV)

            if len(peak_dirs) == 0:
                ax.text(0.5, 0.5, 'No peaks', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # Sort and add peak directions
            trial_df_sorted = trial_df.sort_values(by='initialHeading', ascending=True)
            trial_df_sorted['peakFiringDir'] = peak_dirs

            # Calculate correlation statistics
            xVal = trial_df_sorted['initialHeading'].values
            yVal = trial_df_sorted['peakFiringDir'].values
            realR, slope, _, sig, pValue = homing_angle_corr_stats(xVal, yVal)

            # Store for aggregate histogram
            displayed_cell_stats.append({
                'cellID': cell_id,
                'display_idx': display_idx,
                'realR': realR,
                'slope': slope,
                'sig': sig,
                'pValue': pValue
            })

            # Plot correlation
            plot_kdeplot(
                ax, trial_df_sorted,
                ylabel='Peak firing dir.' if col_idx == 0 else '',
                xlabel='Initial heading',
                set_ylabel=(col_idx == 0),
                c='#cfbaf0'
            )

        except Exception as e:
            print(f"    Warning: Could not create correlation plot for cell {display_idx+1}: {e}")
            ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    # ===== ROW 2: AGGREGATE HISTOGRAM =====

    # Create subplot spanning all 3 columns
    ax_hist = fig.add_subplot(gs[2, :])  # Spans all columns in row 2

    if aggregate_across_sessions:
        # Aggregate across all sessions in useAble list
        print(f"\nRow 2: Computing aggregate statistics across ALL sessions in useAble list...")

        # Import useAble from setup_project
        from setup_project import useAble

        print(f"  Found {len(useAble)} sessions to process")
        if max_cells_per_session is not None:
            print(f"  Limiting to top {max_cells_per_session} cells per session")

        all_cell_stats_df = compute_aggregate_statistics_all_sessions(
            session_list=useAble,
            condition_to_plot=condition_to_plot,
            pickle_dict_trial_matrix=hdLeverCenteredTrialMatrix,
            pickle_dict_stats=hdLeverCenteredLeftRightHeadingError,
            gc_df=gc,
            res_df=res,
            full_sessions_df=full_all_sessions,
            max_cells_per_session=max_cells_per_session
        )

        stats_description = f"All Sessions ({len(useAble)} sessions, {len(all_cell_stats_df)} cells)"

        if len(all_cell_stats_df) > 0:
            plot_correlation_distribution(
                ax_hist, all_cell_stats_df,
                xlabel='Peak-initial heading corr. (r)',
                ylabel='Grid cells',
                legend=True,
                ylim=max(10, len(all_cell_stats_df) // 5),  # Scale with cell count
                title=f'Population Statistics: {len(all_cell_stats_df)} cells from {len(useAble)} sessions',
                set_ylabel=True
            )
        else:
            ax_hist.text(0.5, 0.5, 'No correlation data available',
                        ha='center', va='center', transform=ax_hist.transAxes,
                        fontsize=GLOBALFONTSIZE + 2)
            ax_hist.set_xticks([])
            ax_hist.set_yticks([])

    else:
        # Single-session aggregation (original behavior)
        print(f"\nRow 2: Computing aggregate statistics for top {n_cells_total} cells from {session_name}...")

        # Calculate correlations for ALL top N cells (not just the 3 displayed)
        all_cell_stats = []
        for cell_idx in range(min(n_cells_total, len(top_cells))):
            cell_id = top_cells[cell_idx]

            try:
                condition_full = f'{condition_to_plot}_dark'
                trial_df = extract_firing_rate_trial_matrix(
                    cell_id, condition_full,
                    hdLeverCenteredTrialMatrix, res, initial_heading_df
                )

                if len(trial_df) == 0:
                    continue

                peak_dirs = get_peak_firing_from_slice(trial_df, convolution=GLOBALCONV)

                if len(peak_dirs) == 0:
                    continue

                trial_df_sorted = trial_df.sort_values(by='initialHeading', ascending=True)
                trial_df_sorted['peakFiringDir'] = peak_dirs

                xVal = trial_df_sorted['initialHeading'].values
                yVal = trial_df_sorted['peakFiringDir'].values
                realR, slope, _, sig, pValue = homing_angle_corr_stats(xVal, yVal)

                all_cell_stats.append({
                    'cellID': cell_id,
                    'cell_idx': cell_idx,
                    'realR': realR,
                    'slope': slope,
                    'sig': sig,
                    'pValue': pValue
                })

                if (cell_idx + 1) % 5 == 0:
                    print(f"  Processed {cell_idx + 1}/{min(n_cells_total, len(top_cells))} cells...")

            except Exception as e:
                print(f"    Warning: Could not process cell {cell_idx+1} ({cell_id}): {e}")
                continue

        print(f"  Successfully processed {len(all_cell_stats)}/{min(n_cells_total, len(top_cells))} cells")

        stats_description = f"{session_name} ({len(all_cell_stats)} cells)"

        if len(all_cell_stats) > 0:
            stats_df = pd.DataFrame(all_cell_stats)

            plot_correlation_distribution(
                ax_hist, stats_df,
                xlabel='Peak-initial heading corr. (r)',
                ylabel='Grid cells',
                legend=True,
                ylim=max(10, len(all_cell_stats) + 5),
                title=f'Aggregate Statistics (Top {len(all_cell_stats)} Cells)',
                set_ylabel=True
            )
        else:
            ax_hist.text(0.5, 0.5, 'No correlation data available',
                        ha='center', va='center', transform=ax_hist.transAxes,
                        fontsize=GLOBALFONTSIZE + 2)
            ax_hist.set_xticks([])
            ax_hist.set_yticks([])

    # Add overall title
    if aggregate_across_sessions:
        fig.suptitle(
            f'Grid Cell Firing Rates - {condition_label} Phase\n'
            f'Rows 0-1: Session {session_name} (cells {[i+1 for i in cells_to_display]} of top {n_cells_total}) | '
            f'Row 2: Population statistics',
            fontsize=GLOBALFONTSIZE + 4, y=0.995
        )
    else:
        fig.suptitle(
            f'Grid Cell Firing Rates - {condition_label} Phase\n'
            f'Session: {session_name} | Displaying cells {[i+1 for i in cells_to_display]} of top {n_cells_total}',
            fontsize=GLOBALFONTSIZE + 4, y=0.995
        )

    plt.tight_layout()

    if output_path:
        print(f"\nSaving figure to: {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    return fig


if __name__ == "__main__":
    # Example usage with new 3×3 layout and cross-session aggregation
    session_name = 'jp486-24032023-0108'

    print("\n" + "="*80)
    print("CREATING GRID CELL FIRING RATE ANALYSIS SORTED BY INITIAL HEADING")
    print("="*80)

    # Output directory
    output_dir = 'E:/GitHub/Peng_et.al_2025_noInt/Peng/Output'
    os.makedirs(output_dir, exist_ok=True)

    # Example 1: Population statistics across all sessions
    # Rows 0-1: Show 3 cells from one session for visualization
    # Row 2: Aggregate statistics from ALL sessions in useAble list
    print("\n\nExample 1: Search phase with cross-session population statistics")
    output_file = f'{output_dir}/firing_rate_search_population_{session_name}.pdf'
    fig = create_multi_cell_plot(
        session_name,
        condition_to_plot='searchToLeverPath',
        n_cells_total=10,  # Select from top 10 cells in session_name
        cells_to_display=[1, 5, 6],  # Show top 3 cells in rows 0-1
        aggregate_across_sessions=True,  # NEW: Aggregate across all sessions
        max_cells_per_session=20,  # NEW: Use top 20 cells from each session
        output_path=output_file,
        height_per_row=4
    )

    # Example 2: Single-session mode (original behavior)
    # All statistics from one session only
    # Uncomment to run:
    # print("\n\nExample 2: Single-session mode (no cross-session aggregation)")
    # output_file = f'{output_dir}/firing_rate_search_single_{session_name}.pdf'
    # fig = create_multi_cell_plot(
    #     session_name,
    #     condition_to_plot='searchToLeverPath',
    #     n_cells_total=40,
    #     cells_to_display=[0, 5, 9],
    #     aggregate_across_sessions=False,  # Only this session
    #     output_path=output_file,
    #     height_per_row=4
    # )

    # Example 3: At lever phase with population statistics
    # Uncomment to run:
    # print("\n\nExample 3: At lever phase with population statistics")
    # output_file = f'{output_dir}/firing_rate_lever_population_{session_name}.pdf'
    # fig = create_multi_cell_plot(
    #     session_name,
    #     condition_to_plot='atLever',
    #     n_cells_total=20,
    #     cells_to_display=[0, 2, 4],
    #     aggregate_across_sessions=True,
    #     max_cells_per_session=15,
    #     output_path=output_file,
    #     height_per_row=4
    # )

    # Example 4: Homing phase with population statistics
    # Uncomment to run:
    # print("\n\nExample 4: Homing phase with population statistics")
    # output_file = f'{output_dir}/firing_rate_homing_population_{session_name}.pdf'
    # fig = create_multi_cell_plot(
    #     session_name,
    #     condition_to_plot='homingFromLeavingLeverToPeriphery',
    #     n_cells_total=15,
    #     cells_to_display=[1, 3, 5],  # 2nd, 4th, 6th cells
    #     output_path=output_file,
    #     height_per_row=4
    # )

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nFigure saved to: {output_file}")
    print("\nTo generate other combinations, uncomment Example 2 or 3 above,")
    print("or modify the parameters to your needs:")
    print("\n  Parameters:")
    print("  - condition_to_plot: 'searchToLeverPath', 'atLever', 'homingFromLeavingLeverToPeriphery'")
    print("  - n_cells_total: Total number of top MVL cells for aggregate statistics (e.g., 10, 20, 50)")
    print("  - cells_to_display: List of 3 indices [0-based] to display (e.g., [0,1,2] or [0,5,9])")
    print("  - height_per_row: Height in inches per row (default: 4)")
    print("\n  Figure Layout (3×3 grid):")
    print("    Row 0: Trial matrices for 3 selected cells")
    print("    Row 1: Correlation plots for 3 selected cells")
    print("    Row 2: Aggregate histogram for ALL top N cells (spans full width)")
