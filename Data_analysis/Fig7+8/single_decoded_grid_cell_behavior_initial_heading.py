"""
Script to plot decoded grid cell behavior sorted by initial heading.

This script creates trial matrices showing decoded directional error on single trials
during search (left), at lever (middle), and homing (right) phases. Trials are sorted
by the initial heading of the mouse as it leaves the home base at the start of the trial
(average of first few seconds of search phase).

Based on: single_decoded_grid_cell_behavior.ipynb
Modified to sort by initial heading instead of homing heading.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import pickle
import math
from scipy import stats
from scipy.stats import pearsonr, gaussian_kde
from tqdm import tqdm
import sys
import os
from astropy.stats import circcorrcoef
from sklearn.linear_model import LinearRegression

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Import project setup
exec(open('E:/GitHub/Peng_et.al_2025_noInt/setup_project.py').read())
exec(open('E:/GitHub/Peng_et.al_2025_noInt/generic_plot_functions.py').read())

# ============================================================================
# CONFIGURATION
# ============================================================================

GLOBALFONTSIZE = 12
GLOBALCONV = True
GLOBALSPEEDFILTER = True

# Color scheme
colors = ['#1a2a6c', '#b21f1f', '#fdbb2d']
modelCmap = LinearSegmentedColormap.from_list("custom_colormap", colors, N=256)

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

# Load session data with model predictions
print("Loading session data with model predictions...")
fn = myProject.dataPath + '/results/allSessionDf_with_leverVector_and_last_cohort.csv'
allSessionDf = pd.read_csv(fn, index_col=0)
allSessionDf['light'] = allSessionDf['condition'].apply(lambda x: x.split('_')[1])
allSessionDf['cond_noLight'] = allSessionDf['condition'].apply(lambda x: x.split('_')[0])
allSessionDf['mvtDirError'] = np.negative(allSessionDf['mvtDirError'])

# Load open field data for reference
print("Loading open field data...")
dfOf = pd.read_csv(myProject.dataPath + '/results/reconstuctionDFOF.csv', index_col=0)

full_all_sessions = allSessionDf.copy()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def centerAngles(a):
    """Center angles on their circular mean."""
    # b = a - np.arctan2(np.sum(np.sin(a)), np.sum(np.cos(a)))
    # return np.arctan2(np.sin(b), np.cos(b))
    return a


def interpolate_histogram(hist):
    """Interpolate NaN values in histogram using neighboring values."""
    nan_indices = np.isnan(hist)
    indices = np.arange(len(hist))
    hist[nan_indices] = np.interp(indices[nan_indices], indices[~nan_indices], hist[~nan_indices])
    return hist


def smooth_heatmap(heatmap, sigma=1, ksize=6):
    """Smooth heatmap using Gaussian kernel."""
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
    max_vals[max_vals == min_vals] = 1
    normalized_arr = (np_arr - min_vals) / (max_vals - min_vals)
    return normalized_arr


def get_histos_per_trial(inputDf, bin_edges=None, num_bins=36, bin_var='mvtDirError'):
    """Calculate histogram of directional errors per trial."""
    if bin_edges is None:
        degPerBin = 10
        bin_edges = np.arange(-np.pi, np.pi + degPerBin/360*2*np.pi, degPerBin/360*2*np.pi)
    
    inputDf['bins'] = np.digitize(inputDf[bin_var], bin_edges)
    bin_counts = inputDf['bins'].value_counts().sort_index()
    
    counts_array = np.zeros(num_bins, dtype=int)
    for bin_idx in range(1, num_bins + 1):
        count = bin_counts.get(bin_idx, 0)
        counts_array[bin_idx - 1] = count
    
    return counts_array


def filterForConditions(inputDf, lightCondition='dark'):
    """Filter dataframe to only include trials with all three conditions."""
    desired_conditions = [
        f'atLever_{lightCondition}',
        f'searchToLeverPath_{lightCondition}',
        f'homingFromLeavingLeverToPeriphery_{lightCondition}'
    ]
    
    filtered_trials = inputDf.groupby('trialNo')['condition'].apply(
        lambda x: all(c in x.values for c in desired_conditions)
    )
    
    selected_trials = filtered_trials[filtered_trials].index
    result_df = inputDf[inputDf['trialNo'].isin(selected_trials)].copy()
    
    return result_df


# ============================================================================
# INITIAL HEADING CALCULATION
# ============================================================================

def calculate_initial_heading(sessionSlice, time_window=1):
    """
    Calculate initial heading for each trial as average of first few seconds of search.
    
    Parameters:
    -----------
    sessionSlice : DataFrame
        Session data containing trials
    time_window : float
        Time window in seconds to average over (default: 0.5 seconds)
    
    Returns:
    --------
    DataFrame with initialHeading column added
    """
    print(f"Calculating initial heading (averaging first {time_window} seconds of search)...")
    
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
            # Prefer calculating from position changes for more accurate initial heading
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
    
    # Validation: Check if we have non-zero headings
    if len(initial_headings) > 0:
        non_zero_count = np.sum(np.abs(initial_headings) > 0.01)
        print(f"  Found {len(initial_headings)} trials with initial heading data")
        print(f"  Non-zero headings: {non_zero_count}/{len(initial_headings)}")
        print(f"  Initial heading range: [{np.min(initial_headings):.3f}, {np.max(initial_headings):.3f}] radians")
    else:
        print("  WARNING: No initial heading data found!")
    
    return initial_heading_df


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
        - realR: circular correlation coefficient
        - slope: slope of linear regression
        - meanShuffleR: mean of shuffled correlations
        - significant: boolean indicating if correlation is significant
        - pValue: two-tailed p-value from permutation test
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


def get_circular_mean_shift_rad(hist, fromCenter=True, convolution=False, shiftedMeanArray=None, gaus=True):
    """
    Calculate circular mean shift in radians from histogram.
    
    Parameters:
    -----------
    hist : array-like
        Histogram values
    fromCenter : bool
        Whether to calculate from center (default: True)
    convolution : bool
        Whether to apply convolution (default: False)
    shiftedMeanArray : array-like, optional
        Reference array for convolution
    gaus : bool
        Whether to use Gaussian smoothing (default: True)
    
    Returns:
    --------
    float : circular mean shift in radians
    """
    # Apply convolution or smoothing if required
    if convolution:
        if gaus:
            hist = smooth_heatmap(hist)
        elif shiftedMeanArray is not None:
            hist = np.correlate(hist, shiftedMeanArray, mode='same')
        else:
            raise ValueError("shiftedMeanArray must be provided if gaus is False.")
    
    # Number of bins
    num_bins = len(hist)
    
    # Calculate the width of each bin
    bin_width = 2 * np.pi / num_bins
    
    # Generate the angle for each bin center
    angles = np.linspace(-np.pi, np.pi - bin_width, num_bins)
    
    # Compute the sine and cosine components weighted by the histogram counts
    sin_sum = np.sum(hist * np.sin(angles))
    cos_sum = np.sum(hist * np.cos(angles))
    
    # Calculate the circular mean using atan2 to get the correct quadrant
    circular_mean = math.atan2(sin_sum, cos_sum)
    
    # Normalize the result to be within [-pi, pi]
    circular_mean = math.remainder(circular_mean, 2 * math.pi)
    
    return circular_mean


def get_trial_model_shift(combinedDf, res=None, lightCondition='dark', sesName=''):
    """
    Get trial drift from model predictions for each phase.
    
    Parameters:
    -----------
    combinedDf : DataFrame
        Combined dataframe with trial data
    res : DataFrame, optional
        Results dataframe
    lightCondition : str
        Light condition to filter (default: 'dark')
    sesName : str
        Session name
    
    Returns:
    --------
    DataFrame with trial drift for each phase
    """
    if res is None:
        res = globals()['res']
    
    searchHolder = []
    homingHolder = []
    homingToSearchHolder = []
    trialHolder = []
    searchRadHolder = []
    atLeverRadHolder = []
    homingRadHolder = []
    
    dfOfSlice = dfOf[dfOf.session == sesName].copy()
    histosFromOf = get_histos_per_trial(dfOfSlice)
    combinedDf = filterForConditions(combinedDf, lightCondition)
    
    for trial in combinedDf[combinedDf.light == lightCondition].trialNo.unique():
        trialSlice = combinedDf[combinedDf.trialNo == trial].copy()
        
        # Search phase
        plotSlice = trialSlice[trialSlice.condition == f'searchToLeverPath_{lightCondition}'].copy()
        hist_search = plotSlice.iloc[:, -36:].values.squeeze()
        searchRad = get_circular_mean_shift_rad(hist_search, fromCenter=True, convolution=GLOBALCONV, shiftedMeanArray=histosFromOf)
        
        # At lever phase
        plotSlice = trialSlice[trialSlice.condition == f'atLever_{lightCondition}'].copy()
        hist_lever = plotSlice.iloc[:, -36:].values.squeeze()
        atLeverRad = get_circular_mean_shift_rad(hist_lever, fromCenter=True, convolution=GLOBALCONV, shiftedMeanArray=histosFromOf)
        
        # Homing phase
        plotSlice = trialSlice[trialSlice.condition == f'homingFromLeavingLeverToPeriphery_{lightCondition}'].copy()
        hist_homing = plotSlice.iloc[:, -36:].values.squeeze()
        homingRad = get_circular_mean_shift_rad(hist_homing, fromCenter=True, convolution=GLOBALCONV, shiftedMeanArray=histosFromOf)
        
        searchRadHolder.append(searchRad)
        atLeverRadHolder.append(atLeverRad)
        homingRadHolder.append(homingRad)
        trialHolder.append(trial)
    
    trialModelShift = pd.DataFrame({
        'trialNo': trialHolder,
        'searchRad': searchRadHolder,
        'atLeverRad': atLeverRadHolder,
        'homingRad': homingRadHolder
    })
    
    trialModelShift['sessionName'] = sesName
    trialModelShift['light'] = lightCondition
    
    trialModelShift = pd.merge(trialModelShift, res, how='left', on=['sessionName', 'light', 'trialNo'])
    
    return trialModelShift


def get_circular_stats_dataFrame(sessionName, inputDf, res=None, lightCondition='dark', speedFilter=GLOBALSPEEDFILTER):
    """
    Get circular statistics dataframe with initial heading and trial drift.
    
    Parameters:
    -----------
    sessionName : str
        Session name
    inputDf : DataFrame
        Input dataframe with session data
    res : DataFrame, optional
        Results dataframe
    lightCondition : str
        Light condition (default: 'dark')
    speedFilter : bool
        Whether to filter by speed (default: GLOBALSPEEDFILTER)
    
    Returns:
    --------
    DataFrame with circular statistics including initial heading
    """
    if res is None:
        res = globals()['res']
    
    sessionSlice = inputDf[inputDf.session == sessionName].copy().reset_index()
    
    if speedFilter:
        sessionSlice = sessionSlice[(sessionSlice.speed > 10)].copy()
    
    # Calculate initial heading
    initial_heading_df = calculate_initial_heading(sessionSlice)
    
    combinedDf = get_combinedDf(sessionSlice, res=res.copy(), sesName=sessionName)
    combinedDf = combinedDf.dropna().copy()
    
    trialModelShift = get_trial_model_shift(combinedDf, res.copy(), lightCondition=lightCondition, sesName=sessionName)
    
    # Merge with initial heading
    trialModelShift = pd.merge(trialModelShift, initial_heading_df, on='trialNo', how='left')
    
    trialModelShift = trialModelShift.dropna()
    
    return trialModelShift


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plotRegressionLine(ax, x, y, regColor='#ff9e00'):
    """
    Plot regression line for circular correlation.
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    x : array-like
        X values (angles)
    y : array-like
        Y values (angles)
    regColor : str
        Color for regression line (default: '#ff9e00')
    """
    anglesCentered = centerAngles(x)
    model = LinearRegression()
    model.fit(anglesCentered[:, np.newaxis], y)
    
    # Extend the range of x-values for the line
    x_line = np.array(np.linspace(x.min() - 1, x.max() + 1, 100))
    
    # Generate the y-values for the line using the model
    y_line = model.predict(x_line[:, np.newaxis])
    
    # Plot the regression line
    ax.plot(x_line, y_line, color=regColor, lw=2.5, linestyle='-')


def plot_kdeplot(ax, inputDf, xlim=(0, np.pi), ylim=(0, 1.5), c='#12c2e9', 
                 ylabel='Initial heading', xlabel='Decoded trial drift', 
                 var_x='searchRad', var_y='initialHeading', set_ylabel=True, signLevel=0.025):
    """
    Plot KDE plot with scatter and regression line for circular correlation.
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    inputDf : DataFrame
        Input dataframe
    xlim : tuple
        X-axis limits (default: (0, np.pi))
    ylim : tuple
        Y-axis limits (default: (0, 1.5))
    c : str
        Color for plot (default: '#12c2e9')
    ylabel : str
        Y-axis label (default: 'Initial heading')
    xlabel : str
        X-axis label (default: 'Decoded trial drift')
    var_x : str
        Variable name for x-axis (default: 'searchRad')
    var_y : str
        Variable name for y-axis (default: 'initialHeading')
    set_ylabel : bool
        Whether to set y-axis label (default: True)
    signLevel : float
        Significance level (default: 0.025)
    """
    xVal = inputDf[var_x].values
    yVal = inputDf[var_y].values
    
    # Create 2D KDE plot using matplotlib and scipy
    try:
        # Calculate the point density
        xy = np.vstack([xVal, yVal])
        z = gaussian_kde(xy)(xy)
        
        # Sort points by density for better visualization
        idx = z.argsort()
        x_sorted, y_sorted, z_sorted = xVal[idx], yVal[idx], z[idx]
        
        # Create scatter plot with density colors
        ax.scatter(x_sorted, y_sorted, c=z_sorted, s=40, cmap='Blues', 
                  edgecolor='black', linewidth=1, alpha=0.8)
    except:
        # Fallback to simple scatter if KDE fails
        ax.scatter(xVal, yVal, s=40, edgecolor='black', linewidth=1, 
                  color=c, alpha=0.8)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plotRegressionLine(ax, x=xVal, y=yVal, regColor='#ff9e00')
    
    if set_ylabel:
        ax.set_yticks(ticks=[-np.pi, 0, np.pi])
        ax.set_yticklabels(["-$\pi$", "0", "$\pi$"], fontsize=GLOBALFONTSIZE)
        ax.set_ylabel(ylabel, fontsize=GLOBALFONTSIZE, labelpad=1)
    else:
        ax.set_yticks([])
        ax.set_yticklabels([], fontsize=GLOBALFONTSIZE)
        ax.set_ylabel('', fontsize=GLOBALFONTSIZE)
    
    ax.set_xticks(ticks=[-np.pi, 0, np.pi])
    ax.set_xticklabels(["-$\pi$", "0", "$\pi$"], fontsize=GLOBALFONTSIZE)
    
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


def plot_correlation_distribution(ax, accumulated_stats_df, phase_var='searchRad',
                                   xlabel='Drift-initial heading corr. (r)',
                                   ylabel='Sessions', legend=True, ylim=50, 
                                   title='', set_ylabel=True):
    """
    Plot histogram of circular correlation coefficients accumulated across sessions.
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    accumulated_stats_df : DataFrame
        DataFrame with columns: 'realR', 'sig', 'phase_var'
        Contains accumulated correlation statistics from all sessions up to current
    phase_var : str
        Which phase variable to filter for ('searchRad', 'atLeverRad', 'homingRad')
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    legend : bool
        Whether to show legend
    ylim : float
        Y-axis upper limit
    title : str
        Plot title
    set_ylabel : bool
        Whether to set y-axis label
    """
    # Filter for the specific phase
    phase_data = accumulated_stats_df[accumulated_stats_df['phase_var'] == phase_var].copy()
    
    if len(phase_data) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    
    # Map boolean to string labels
    phase_data['sig_label'] = phase_data['sig'].map({True: 'Significant', False: 'Non sign.'})
    
    # Color palette
    palette = {'Significant': '#1a659e', 'Non sign.': '#ff6b35'}
    order = ['Significant', 'Non sign.']
    
    # Create histogram
    sns.histplot(
        data=phase_data, x='realR', hue='sig_label', 
        bins=10, kde=False, palette=palette, 
        legend=legend, hue_order=order, ax=ax
    )
    
    # Styling
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
    realR_values = phase_data['realR'].values
    if len(realR_values) > 0:
        stat, p_value = stats.wilcoxon(realR_values)
        
        # Display statistics
        n_sessions = len(phase_data)
        ax.text(0.07, 0.9, f'N = {n_sessions}', fontsize=GLOBALFONTSIZE, transform=ax.transAxes)
        
        # Format p-value
        if p_value < 0.0001:
            p_text = 'p < 0.0001'
        else:
            p_text = f'p = {round(p_value, 4)}'
        ax.text(0.07, 0.78, p_text, fontsize=GLOBALFONTSIZE, transform=ax.transAxes, fontweight='bold')
    
    # Adjust legend
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles, labels, fontsize=GLOBALFONTSIZE - 1, 
            frameon=False, loc='lower left', 
            bbox_to_anchor=(-0.01, 0.65)
        )
    
    ax.tick_params(axis='both', which='both', labelsize=GLOBALFONTSIZE)



def get_trial_drift_from_slice_ls(inputDf, getMatrix=False, shiftedMeanArray=None, 
                                   sortCondition='', convolution=True, gaus=True):
    """Get trial drift from slice, optionally returning full matrix."""
    if len(sortCondition) == 0:
        sortSlice = inputDf.copy()
    else:
        sortSlice = inputDf.sort_values(by=sortCondition, ascending=True)
    
    corArrayHolder = []
    interHolder = []
    
    shiftIndex = np.argmax(np.nanmean(sortSlice.iloc[:, -37:-1], axis=0))
    
    for i in range(0, sortSlice.shape[0]):
        heatmapSlice = np.array(sortSlice.iloc[i, -37:-1]).astype(float)
        heatmapSlice = interpolate_histogram(heatmapSlice)
        heatmapSlice = np.roll(heatmapSlice, 18 - shiftIndex)
        
        if gaus:
            crossCor = smooth_heatmap(heatmapSlice)
        else:
            crossCor = np.correlate(heatmapSlice, shiftedMeanArray, mode='same')
        
        corArrayHolder.append(crossCor)
        interHolder.append(heatmapSlice)
    
    if not convolution:
        corArrayHolder = interHolder
    
    if getMatrix:
        return corArrayHolder, np.argmax(corArrayHolder, axis=1)
    
    driftIndex = np.argmax(corArrayHolder, axis=1) - 18
    shiftDegree = driftIndex * 10
    radians = np.radians(shiftDegree)
    
    def wrap_radians(arr, divisor=math.tau):
        r_func = np.vectorize(lambda x: math.remainder(x, divisor))
        return r_func(arr)
    
    radians = wrap_radians(radians)
    radians = centerAngles(radians)
    
    return radians


def get_combinedDf(sessionSlice, res=None, sesName=''):
    """Get combined dataframe with histograms for each condition and trial."""
    if res is None:
        res = globals()['res']
    
    condHolder = []
    trialHolder = []
    mvlHolder = []
    histosHolder = []
    
    for cond in sessionSlice.condition.unique():
        for t in sessionSlice[sessionSlice.condition == cond].trialNo.unique():
            sliceDf = sessionSlice[(sessionSlice.condition == cond) & (sessionSlice.trialNo == t)].copy()
            histos = get_histos_per_trial(sliceDf)
            histosHolder.append(histos)
            
            complex_data = np.exp(1j * sliceDf['mvtDirError'])
            mean_vector = np.mean(complex_data)
            mean_vector_length = np.abs(mean_vector)
            
            condHolder.append(cond)
            trialHolder.append(t)
            mvlHolder.append(mean_vector_length)
    
    histosDf = pd.DataFrame(histosHolder, columns=[f'bin_{i}' for i in range(1, 37)])
    
    combinedDf = pd.DataFrame({
        'condition': condHolder,
        'trialNo': trialHolder,
        'mvl': mvlHolder
    })
    combinedDf['sessionName'] = sesName
    combinedDf['light'] = combinedDf.condition.apply(lambda x: x.split('_')[1])
    combinedDf = pd.merge(combinedDf, res, how='left', on=['sessionName', 'trialNo', 'light'])
    
    combinedDf = pd.concat([combinedDf, histosDf], axis=1)
    combinedDf = combinedDf[combinedDf.valid == True]
    
    return combinedDf


def plot_model_prediction_heatmap(ax, sesName, condition='atLever_dark', 
                                   set_ticks=False, ylabel='', xlabel='Dec. dir. error',
                                   sortBy='initialHeading'):
    """
    Plot model prediction heatmap sorted by initial heading.
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    sesName : str
        Session name
    condition : str
        Condition to plot
    set_ticks : bool
        Whether to show circular mean markers
    ylabel : str
        Y-axis label
    xlabel : str
        X-axis label
    sortBy : str
        Column to sort by (default: 'initialHeading')
    """
    # Get session data
    sessionSlice = full_all_sessions[full_all_sessions.session == sesName].reset_index()
    if GLOBALSPEEDFILTER:
        sessionSlice = sessionSlice[sessionSlice.speed > 10]
    
    # Calculate initial heading for this session
    initial_heading_df = calculate_initial_heading(sessionSlice)
    
    # Get combined dataframe
    combinedDf = get_combinedDf(sessionSlice.copy(), res=res.copy(), sesName=sesName)
    combinedDf = filterForConditions(combinedDf).copy()
    
    # Merge with initial heading
    combinedDf = pd.merge(combinedDf, initial_heading_df, on='trialNo', how='left')
    
    # Get reference histograms from open field
    dfOfSlice = dfOf[dfOf.session == sesName].copy()
    histosFromOf = get_histos_per_trial(dfOfSlice)
    
    # Filter for condition and sort by initial heading
    inputLS = combinedDf[combinedDf.condition == condition].copy()
    inputLS = inputLS.sort_values(by=sortBy, ascending=True)
    
    # Get trial drift matrix
    plotArray, maxInd = get_trial_drift_from_slice_ls(
        inputLS, getMatrix=True, shiftedMeanArray=histosFromOf,
        sortCondition=sortBy, convolution=GLOBALCONV
    )
    
    plotArray = normalize_2d_array(plotArray)
    
    # Plot heatmap
    sns.heatmap(plotArray, cmap=modelCmap, cbar=False, ax=ax)
    
    ax.set_xticks(ticks=[0, 18, 36])
    ax.set_xticklabels(["-$\pi$", "0", "$\pi$"], fontsize=GLOBALFONTSIZE)
    
    if set_ticks:
        for i, max_index in enumerate(maxInd):
            sns.scatterplot(
                x=[max_index + 0.5], y=[i + 0.5], ax=ax,
                marker='o', color='white', s=30, edgecolor='black', linewidth=1
            )
    
    ax.set_yticks([])
    ax.set_yticklabels([])
    
    ax.set_ylabel(ylabel, fontsize=GLOBALFONTSIZE)
    ax.set_xlabel(xlabel, fontsize=GLOBALFONTSIZE)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def save_session_info(session_list, output_path):
    """
    Save session information to a text file.
    
    Parameters:
    -----------
    session_list : list
        List of session names
    output_path : str
        Path to save the text file
    """
    from datetime import datetime
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DECODED DIRECTIONAL ERROR - MULTI-SESSION PLOT\n")
        f.write("Sorted by Initial Heading\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Number of sessions: {len(session_list)}\n\n")
        
        f.write("Sessions included:\n")
        f.write("-"*80 + "\n")
        for i, session in enumerate(session_list, 1):
            f.write(f"{i}. {session}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Configuration:\n")
        f.write("-"*80 + "\n")
        f.write(f"Global Font Size: {GLOBALFONTSIZE}\n")
        f.write(f"Convolution: {GLOBALCONV}\n")
        f.write(f"Speed Filter: {GLOBALSPEEDFILTER}\n")
        f.write(f"Height per session: 4 inches\n")
        f.write("="*80 + "\n")
    
    print(f"Session info saved to: {output_path}")


def create_multi_session_plot(session_list, output_path=None, height_per_session=4):
    """
    Create multi-session plot with (3N)×3 subplots showing search, at lever, and homing phases
    sorted by initial heading. Each session has three rows: trial matrices, correlation plots,
    and correlation distribution histograms.
    
    Parameters:
    -----------
    session_list : list of str
        List of session names to plot
    output_path : str, optional
        Path to save figure. If None, displays instead.
    height_per_session : float, optional
        Height in inches per session row triplet (default: 4)
    """
    n_sessions = len(session_list)
    print(f"\nCreating multi-session plot for {n_sessions} session(s)")
    
    # Calculate figure size - now we have 3 rows per session
    fig_width = 12
    fig_height = 3 * n_sessions * height_per_session
    
    # Create figure with (3N)×3 grid
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(3 * n_sessions, 3, figure=fig, wspace=0.3, hspace=0.5)
    
    # Initialize accumulator for correlation statistics
    accumulated_stats = []
    
    # Loop through sessions and create 9 panels per session (3 rows × 3 columns)
    for session_idx, session_name in enumerate(session_list):
        print(f"  Processing session {session_idx + 1}/{n_sessions}: {session_name}")
        
        # Calculate row indices for this session
        heatmap_row = 3 * session_idx
        correlation_row = 3 * session_idx + 1
        histogram_row = 3 * session_idx + 2
        
        # ===== FIRST ROW: TRIAL MATRICES =====
        
        # Left panel: Search heatmap
        ax0 = fig.add_subplot(gs[heatmap_row, 0])
        plot_model_prediction_heatmap(
            ax0, session_name, 'searchToLeverPath_dark',
            set_ticks=True, 
            ylabel=f'{session_name}\nTrials' if session_idx == 0 else session_name,
            xlabel=''
        )
        if session_idx == 0:
            ax0.set_title('Search', fontsize=GLOBALFONTSIZE + 2)
        
        # Middle panel: At lever heatmap
        ax1 = fig.add_subplot(gs[heatmap_row, 1])
        plot_model_prediction_heatmap(
            ax1, session_name, 'atLever_dark',
            set_ticks=True, 
            ylabel='',
            xlabel=''
        )
        if session_idx == 0:
            ax1.set_title('At Lever', fontsize=GLOBALFONTSIZE + 2)
        
        # Right panel: Homing heatmap
        ax2 = fig.add_subplot(gs[heatmap_row, 2])
        plot_model_prediction_heatmap(
            ax2, session_name, 'homingFromLeavingLeverToPeriphery_dark',
            set_ticks=True, 
            ylabel='',
            xlabel=''
        )
        if session_idx == 0:
            ax2.set_title('Homing', fontsize=GLOBALFONTSIZE + 2)
        
        # ===== SECOND ROW: CORRELATION PLOTS =====
        
        # Get circular statistics dataframe with initial heading
        try:
            c_statsDf = get_circular_stats_dataFrame(session_name, allSessionDf.copy(), res.copy())
            
            # Calculate correlation statistics for each phase
            phase_vars = ['searchRad', 'atLeverRad', 'homingRad']
            for phase_var in phase_vars:
                xVal = c_statsDf[phase_var].values
                yVal = c_statsDf['initialHeading'].values
                realR, slope, _, sig, pValue = homing_angle_corr_stats(xVal, yVal)
                
                # Add to accumulator
                accumulated_stats.append({
                    'sessionName': session_name,
                    'phase_var': phase_var,
                    'realR': realR,
                    'slope': slope,
                    'sig': sig,
                    'pValue': pValue
                })
            
            # Left panel: Search correlation
            ax3 = fig.add_subplot(gs[correlation_row, 0])
            plot_kdeplot(
                ax3, c_statsDf, 
                ylabel='Initial heading', 
                xlabel='',
                var_x='searchRad', 
                var_y='initialHeading', 
                c='#cfbaf0'
            )
            
            # Middle panel: At lever correlation
            ax4 = fig.add_subplot(gs[correlation_row, 1])
            plot_kdeplot(
                ax4, c_statsDf, 
                ylabel='Initial heading', 
                xlabel='',
                var_x='atLeverRad', 
                var_y='initialHeading', 
                set_ylabel=False, 
                c='#cfbaf0'
            )
            
            # Right panel: Homing correlation
            ax5 = fig.add_subplot(gs[correlation_row, 2])
            plot_kdeplot(
                ax5, c_statsDf, 
                ylabel='Initial heading', 
                xlabel='',
                var_x='homingRad', 
                var_y='initialHeading', 
                set_ylabel=False, 
                c='#cfbaf0'
            )
            
        except Exception as e:
            print(f"    Warning: Could not create correlation plots for {session_name}: {e}")
            # Create empty plots if correlation calculation fails
            for col_idx in range(3):
                ax = fig.add_subplot(gs[correlation_row, col_idx])
                ax.text(0.5, 0.5, 'Data unavailable', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
        
        # ===== THIRD ROW: CORRELATION DISTRIBUTION HISTOGRAMS =====
        
        # Convert accumulated stats to DataFrame
        if len(accumulated_stats) > 0:
            accumulated_df = pd.DataFrame(accumulated_stats)
            
            # Left panel: Search histogram
            ax6 = fig.add_subplot(gs[histogram_row, 0])
            plot_correlation_distribution(
                ax6, accumulated_df, 
                phase_var='searchRad',
                xlabel='Drift-initial heading corr. (r)' if session_idx == n_sessions - 1 else '',
                ylabel='Sessions',
                legend=(session_idx == 0),
                ylim=n_sessions + 5,
                title=''
            )
            
            # Middle panel: At lever histogram
            ax7 = fig.add_subplot(gs[histogram_row, 1])
            plot_correlation_distribution(
                ax7, accumulated_df, 
                phase_var='atLeverRad',
                xlabel='Drift-initial heading corr. (r)' if session_idx == n_sessions - 1 else '',
                ylabel='',
                legend=False,
                ylim=n_sessions + 5,
                title='',
                set_ylabel=False
            )
            
            # Right panel: Homing histogram
            ax8 = fig.add_subplot(gs[histogram_row, 2])
            plot_correlation_distribution(
                ax8, accumulated_df, 
                phase_var='homingRad',
                xlabel='Drift-initial heading corr. (r)' if session_idx == n_sessions - 1 else '',
                ylabel='',
                legend=False,
                ylim=n_sessions + 5,
                title='',
                set_ylabel=False
            )
        else:
            # Create empty histogram plots
            for col_idx in range(3):
                ax = fig.add_subplot(gs[histogram_row, col_idx])
                ax.text(0.5, 0.5, 'No correlation data', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
    
    # Add overall title
    fig.suptitle(
        f'Decoded Directional Error Sorted by Initial Heading\n{n_sessions} Session(s)',
        fontsize=GLOBALFONTSIZE + 4, y=0.995
    )
    
    plt.tight_layout()
    
    if output_path:
        print(f"Saving figure to: {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        # Save session info file
        info_path = output_path.rsplit('.', 1)[0] + '_session_info.txt'
        save_session_info(session_list, info_path)
    else:
        plt.show()
    
    return fig


if __name__ == "__main__":
    # Example usage: Define list of sessions to plot
    # Each session will be displayed as a separate row in the figure
    
    # Example with a single session (produces 1×3 subplot grid)
    # session_list = ['jp486-05032023-0108']
    
    # Example with multiple sessions (produces N×3 subplot grid)
    session_list = ['jp486-19032023-0108', 'jp486-18032023-0108',
       'jp3269-28112022-0108', 'jp486-16032023-0108',
       'jp452-25112022-0110', 'jp486-24032023-0108',
       'jp486-22032023-0108', 'jp452-24112022-0109',
       'jp486-15032023-0108', 'jp3120-25052022-0107',
       'jp3120-26052022-0107', 'jp451-28102022-0108',
       'jp486-20032023-0108', 'jp486-06032023-0108',
       'jp486-26032023-0108', 'jp486-17032023-0108',
       'jp451-29102022-0108', 'jp451-30102022-0108',
       'jp486-10032023-0108', 'jp486-05032023-0108',
       'jp3269-29112022-0108', 'mn8578-17122021-0107',
       'jp452-23112022-0108', 'jp1686-26042022-0108']
    
    print("\n" + "="*80)
    print("CREATING MULTI-SESSION TRIAL MATRIX SORTED BY INITIAL HEADING")
    print("="*80)
    
    # Create the multi-session plot
    output_dir = 'E:/GitHub/Peng_et.al_2025_noInt/Peng/Output'
    output_file = f'{output_dir}/trial_matrix_initial_heading_multi_session.pdf'
    
    fig = create_multi_session_plot(
        session_list,
        output_path=output_file,
        height_per_session=4
    )
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nFigure saved to: {output_file}")
    print(f"Session info saved to: {output_file.replace('.pdf', '_session_info.txt')}")
    print("\nTo plot different sessions, modify the 'session_list' variable.")
