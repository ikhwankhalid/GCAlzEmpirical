"""
Turn-Based Heading Deviation CHANGE Analysis

This script analyzes **change in heading deviation** patterns around turning events to investigate
whether the direction of a turn (left vs right) results in systematic differences in how quickly
animals correct or accumulate heading errors.

Analysis Pipeline:
1. Detect left and right turns based on heading changes
2. Extract heading deviation data ±1 second around each turn
3. **Calculate rate of change of heading deviation (d(mvtDirError)/dt)**
4. Align all turns to t=0 and separate by turn direction
5. Plot individual traces and mean responses to identify correction dynamics

Key Difference from Original:
- **Original**: Analyzes absolute heading deviation (mvtDirError)
- **This script**: Analyzes **rate of change** of heading deviation (rad/s)
  - Positive values = heading error increasing (drifting away)
  - Negative values = heading error decreasing (correcting)
  - Zero = heading error stable

Data Source:
- Uses the same reconstruction data as `heading_deviation_by_time.ipynb`
- Analyzes `mvtDirError` (heading deviation) from `reconstuctionDFAutoPI.csv`
"""

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.ndimage
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100

# Setup paths
PROJECT_DATA_PATH = 'E:\\GitHub\\Peng_et.al_2025_noInt\\Peng'

# Sessions to use
useAble = ['jp486-19032023-0108', 'jp486-18032023-0108',
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


def calculate_heading_from_position(x, y, smooth_window=5.):
    """Calculate instantaneous heading from position data."""
    if smooth_window > 1:
        x_smooth = scipy.ndimage.gaussian_filter1d(x, sigma=smooth_window/3)
        y_smooth = scipy.ndimage.gaussian_filter1d(y, sigma=smooth_window/3)
    else:
        x_smooth, y_smooth = x, y
    
    dx = np.diff(x_smooth)
    dy = np.diff(y_smooth)
    
    heading = np.full(len(x), np.nan)
    heading[1:] = np.arctan2(dy, dx)
    
    return heading


def detect_turns(heading, time, min_turn_threshold=np.pi/12, min_time_between_turns=0.5):
    """Detect left and right turns from heading data."""
    dheading = np.diff(heading)
    dheading = np.arctan2(np.sin(dheading), np.cos(dheading))
    
    turn_indices = np.where(np.abs(dheading) >= min_turn_threshold)[0] + 1
    
    if len(turn_indices) == 0:
        return []
    
    turns_info = []
    last_turn_time = -np.inf
    
    for idx in turn_indices:
        if idx < len(time) and not np.isnan(heading[idx]):
            turn_time = time[idx]
            
            if turn_time - last_turn_time >= min_time_between_turns:
                turn_magnitude = dheading[idx-1]
                turn_direction = 'left' if turn_magnitude > 0 else 'right'
                
                turns_info.append({
                    'time': turn_time,
                    'index': idx,
                    'direction': turn_direction,
                    'magnitude': turn_magnitude
                })
                
                last_turn_time = turn_time
    
    return turns_info


def calculate_heading_deviation_change(heading_deviation, time, smooth_sigma=2.0):
    """
    Calculate the rate of change of heading deviation.
    
    Parameters:
    -----------
    heading_deviation : array
        Heading deviation values (rad)
    time : array
        Time values (s)
    smooth_sigma : float
        Sigma for Gaussian smoothing before differentiation (default=2.0)
    
    Returns:
    --------
    heading_dev_change : array
        Rate of change of heading deviation (rad/s)
    time_midpoints : array
        Midpoint times for the change values
    """
    # Smooth heading deviation to reduce noise before differentiation
    heading_dev_smooth = scipy.ndimage.gaussian_filter1d(
        heading_deviation, sigma=smooth_sigma, mode='nearest'
    )
    
    dheading_dev = np.diff(heading_dev_smooth)
    dt = np.diff(time)
    
    heading_dev_change = np.where(dt != 0, dheading_dev / dt, np.nan)
    time_midpoints = time[:-1] + dt / 2
    
    return heading_dev_change, time_midpoints


def extract_heading_deviation_around_turns(trial_data, turns_info, window_seconds=1.0):
    """Extract heading deviation data and its CHANGE around turn events."""
    turn_aligned_data = []
    
    for turn in turns_info:
        turn_time = turn['time']
        
        time_mask = (trial_data['recTime'] >= turn_time - window_seconds) & \
                   (trial_data['recTime'] <= turn_time + window_seconds)
        
        if np.sum(time_mask) > 0:
            window_data = trial_data[time_mask].copy()
            
            time_rel = window_data['recTime'].values - turn_time
            heading_deviation = window_data['mvtDirError'].values
            
            # Calculate heading deviation change
            if len(heading_deviation) > 1:
                heading_dev_change, time_midpoints = calculate_heading_deviation_change(
                    heading_deviation, window_data['recTime'].values)
                time_rel_change = time_midpoints - turn_time
            else:
                heading_dev_change = np.array([])
                time_rel_change = np.array([])
            
            turn_aligned_data.append({
                'direction': turn['direction'],
                'time_rel': time_rel,
                'heading_deviation': heading_deviation,
                'heading_deviation_change': heading_dev_change,
                'time_rel_change': time_rel_change,
                'magnitude': turn['magnitude'],
                'turn_time': turn_time,
                'session': window_data.iloc[0]['session'] if len(window_data) > 0 else None,
                'trial': window_data.iloc[0]['trial'] if len(window_data) > 0 else None,
                'condition': window_data.iloc[0]['condition'] if len(window_data) > 0 else None
            })
    
    return turn_aligned_data


def process_all_trials_for_turns(df, conditions=['all_light', 'all_dark']):
    """Process all trials to detect turns and extract heading deviation change data."""
    all_turn_data = {}
    
    for condition in conditions:
        print(f"\nProcessing {condition} trials...")
        condition_df = df[df.condition == condition].copy()
        
        condition_df['session_trial'] = condition_df['session'] + '_T' + condition_df['trial'].astype(str)
        unique_trials = condition_df['session_trial'].unique()
        
        print(f"Found {len(unique_trials)} unique trials")
        
        condition_turn_data = {'left': [], 'right': []}
        
        for trial_id in tqdm(unique_trials, desc=f"Processing {condition} trials"):
            trial_data = condition_df[condition_df['session_trial'] == trial_id].copy()
            trial_data = trial_data.sort_values('recTime')
            
            if len(trial_data) < 10:
                continue
            
            heading = calculate_heading_from_position(trial_data['x'].values, trial_data['y'].values)
            turns_info = detect_turns(heading, trial_data['recTime'].values)
            
            if len(turns_info) == 0:
                continue
            
            turn_aligned_data = extract_heading_deviation_around_turns(trial_data, turns_info)
            
            for turn_data in turn_aligned_data:
                direction = turn_data['direction']
                condition_turn_data[direction].append(turn_data)
        
        all_turn_data[condition] = condition_turn_data
        
        print(f"Found {len(condition_turn_data['left'])} left turns and {len(condition_turn_data['right'])} right turns")
    
    return all_turn_data


def interpolate_to_common_timebase(turn_data_list, time_bins, use_change=True):
    """Interpolate all turn data to a common time base for averaging."""
    n_turns = len(turn_data_list)
    n_time_points = len(time_bins)
    interpolated_data = np.full((n_turns, n_time_points), np.nan)
    
    for i, turn_data in enumerate(turn_data_list):
        if use_change:
            time_rel = turn_data['time_rel_change']
            values = turn_data['heading_deviation_change']
        else:
            time_rel = turn_data['time_rel']
            values = turn_data['heading_deviation']
        
        valid_mask = ~np.isnan(values)
        if np.sum(valid_mask) < 3:
            continue
            
        time_valid = time_rel[valid_mask]
        values_valid = values[valid_mask]
        
        time_min, time_max = time_valid.min(), time_valid.max()
        coverage_mask = (time_bins >= time_min) & (time_bins <= time_max)
        
        if np.sum(coverage_mask) > 0:
            try:
                interpolated_values = np.interp(time_bins[coverage_mask], time_valid, values_valid)
                interpolated_data[i, coverage_mask] = interpolated_values
            except:
                continue
    
    return interpolated_data


def plot_turn_aligned_heading_deviation_change(all_turn_data, condition, max_individual_traces=50, 
                                              time_window=1.0, n_time_bins=100):
    """Plot heading deviation CHANGE aligned to turn events."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    colors = {'left': 'red', 'right': 'blue'}
    time_bins = np.linspace(-time_window, time_window, n_time_bins)
    condition_data = all_turn_data[condition]
    
    for col, direction in enumerate(['left', 'right']):
        turn_data_list = condition_data[direction]
        
        if len(turn_data_list) == 0:
            axes[0, col].text(0.5, 0.5, f'No {direction} turns found', 
                            ha='center', va='center', transform=axes[0, col].transAxes)
            axes[1, col].text(0.5, 0.5, f'No {direction} turns found', 
                            ha='center', va='center', transform=axes[1, col].transAxes)
            continue
        
        # Plot individual traces (top row)
        n_traces_to_plot = min(max_individual_traces, len(turn_data_list))
        selected_indices = np.random.choice(len(turn_data_list), n_traces_to_plot, replace=False)
        
        for idx in selected_indices:
            turn_data = turn_data_list[idx]
            axes[0, col].plot(turn_data['time_rel_change'], turn_data['heading_deviation_change'], 
                            color=colors[direction], alpha=0.3, linewidth=0.5)
        
        axes[0, col].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        axes[0, col].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[0, col].set_xlabel('Time from turn (s)')
        axes[0, col].set_ylabel('Heading deviation change (rad/s)')
        axes[0, col].set_title(f'{direction.capitalize()} turns - Individual traces\n(n={len(turn_data_list)}, showing {n_traces_to_plot})')
        axes[0, col].set_xlim(-time_window, time_window)
        axes[0, col].grid(True, alpha=0.3)
        
        # Calculate and plot mean (bottom row)
        interpolated_data = interpolate_to_common_timebase(turn_data_list, time_bins, use_change=True)
        
        # Calculate mean and SEM (using linear statistics for change)
        mean_change = np.nanmean(interpolated_data, axis=0)
        sem_change = np.nanstd(interpolated_data, axis=0) / np.sqrt(np.sum(~np.isnan(interpolated_data), axis=0))
        
        valid_points = ~np.isnan(mean_change)
        axes[1, col].plot(time_bins[valid_points], mean_change[valid_points], 
                        color=colors[direction], linewidth=2, label=f'Mean {direction} turns')
        axes[1, col].fill_between(time_bins[valid_points], 
                                 mean_change[valid_points] - sem_change[valid_points],
                                 mean_change[valid_points] + sem_change[valid_points],
                                 color=colors[direction], alpha=0.2)
        
        axes[1, col].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        axes[1, col].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[1, col].set_xlabel('Time from turn (s)')
        axes[1, col].set_ylabel('Heading deviation change (rad/s)')
        axes[1, col].set_title(f'{direction.capitalize()} turns - Mean ± SEM')
        axes[1, col].set_xlim(-time_window, time_window)
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].legend()
    
    plt.suptitle(f'Turn-Aligned Heading Deviation CHANGE Analysis - {condition.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_left_vs_right_comparison_change(all_turn_data, condition, time_window=1., n_time_bins=100):
    """Plot direct comparison of left vs right turn effects on heading deviation change."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    time_bins = np.linspace(-time_window, time_window, n_time_bins)
    condition_data = all_turn_data[condition]
    colors = {'left': 'red', 'right': 'blue'}
    
    for direction in ['left', 'right']:
        turn_data_list = condition_data[direction]
        
        if len(turn_data_list) == 0:
            continue
        
        interpolated_data = interpolate_to_common_timebase(turn_data_list, time_bins, use_change=True)
        mean_change = np.nanmean(interpolated_data, axis=0)
        sem_change = np.nanstd(interpolated_data, axis=0) / np.sqrt(np.sum(~np.isnan(interpolated_data), axis=0))
        
        valid_points = ~np.isnan(mean_change)
        axes[0].plot(time_bins[valid_points], mean_change[valid_points], 
                    color=colors[direction], linewidth=2, label=f'{direction.capitalize()} turns (n={len(turn_data_list)})')
        axes[0].fill_between(time_bins[valid_points], 
                           mean_change[valid_points] - sem_change[valid_points],
                           mean_change[valid_points] + sem_change[valid_points],
                           color=colors[direction], alpha=0.2)
    
    axes[0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
    axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[0].set_xlabel('Time from turn (s)')
    axes[0].set_ylabel('Heading deviation change (rad/s)')
    axes[0].set_title('Left vs Right Turn Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Calculate and plot difference
    left_data = condition_data['left']
    right_data = condition_data['right']
    
    if len(left_data) > 0 and len(right_data) > 0:
        left_interpolated = interpolate_to_common_timebase(left_data, time_bins, use_change=True)
        right_interpolated = interpolate_to_common_timebase(right_data, time_bins, use_change=True)
        
        left_mean = np.nanmean(left_interpolated, axis=0)
        right_mean = np.nanmean(right_interpolated, axis=0)
        difference = left_mean - right_mean
        
        left_sem = np.nanstd(left_interpolated, axis=0) / np.sqrt(np.sum(~np.isnan(left_interpolated), axis=0))
        right_sem = np.nanstd(right_interpolated, axis=0) / np.sqrt(np.sum(~np.isnan(right_interpolated), axis=0))
        diff_sem = np.sqrt(left_sem**2 + right_sem**2)
        
        valid_points = ~np.isnan(difference)
        axes[1].plot(time_bins[valid_points], difference[valid_points], 
                    color='purple', linewidth=2, label='Left - Right')
        axes[1].fill_between(time_bins[valid_points], 
                           difference[valid_points] - diff_sem[valid_points],
                           difference[valid_points] + diff_sem[valid_points],
                           color='purple', alpha=0.2)
    
    axes[1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1].set_xlabel('Time from turn (s)')
    axes[1].set_ylabel('Heading deviation change difference (rad/s)')
    axes[1].set_title('Difference (Left - Right)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Turn Direction Comparison - CHANGE - {condition.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def main():
    """Main analysis function."""
    print("="*80)
    print("TURN-BASED HEADING DEVIATION CHANGE ANALYSIS")
    print("="*80)
    
    # Load data
    print("\nLoading reconstruction data...")
    fn = os.path.join(PROJECT_DATA_PATH, "results", "reconstuctionDFAutoPI.csv")
    dfAutoPI = pd.read_csv(fn)
    print(f"Loaded {len(dfAutoPI)} rows")
    
    # Filter for useable sessions
    dfAutoPI = dfAutoPI[dfAutoPI.session.isin(useAble)]
    print(f"After filtering: {len(dfAutoPI)} rows, {dfAutoPI.session.nunique()} sessions")
    
    # Process all data
    print("\nProcessing all trials for turn detection...")
    all_turn_data = process_all_trials_for_turns(dfAutoPI)
    
    # Display summary
    print("\n" + "="*60)
    print("TURN DETECTION SUMMARY")
    print("="*60)
    for condition in all_turn_data.keys():
        print(f"\n{condition.upper()}:")
        left_turns = len(all_turn_data[condition]['left'])
        right_turns = len(all_turn_data[condition]['right'])
        print(f"  Left turns: {left_turns}")
        print(f"  Right turns: {right_turns}")
        print(f"  Total turns: {left_turns + right_turns}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Light trials
    fig_light = plot_turn_aligned_heading_deviation_change(all_turn_data, 'all_light')
    output_path = os.path.join(PROJECT_DATA_PATH, 'results', 'turn_aligned_heading_deviation_change_light.png')
    fig_light.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    fig_light_comp = plot_left_vs_right_comparison_change(all_turn_data, 'all_light')
    output_path = os.path.join(PROJECT_DATA_PATH, 'results', 'turn_comparison_change_light.png')
    fig_light_comp.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    # Dark trials
    fig_dark = plot_turn_aligned_heading_deviation_change(all_turn_data, 'all_dark')
    output_path = os.path.join(PROJECT_DATA_PATH, 'results', 'turn_aligned_heading_deviation_change_dark.png')
    fig_dark.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    fig_dark_comp = plot_left_vs_right_comparison_change(all_turn_data, 'all_dark')
    output_path = os.path.join(PROJECT_DATA_PATH, 'results', 'turn_comparison_change_dark.png')
    fig_dark_comp.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    print("\nAnalysis complete!")
    print("="*80)
    
    plt.show()


if __name__ == "__main__":
    main()
