"""
Script to plot heading deviation (mvtDirError) over time for light and dark trials.

Creates two figures:
1. Light trials: Time vs heading deviation
2. Dark trials: Time vs heading deviation

Time is relative to trial start, ylim is (-π, π)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Setup
PROJECT_DATA_PATH = 'E:\\GitHub\\Peng_et.al_2025_noInt\\Peng'

# Sessions to use (from setup_project.py)
useAble = ['jp486-19032023-0108', 'jp486-18032023-0108',
       'jp3269-28112022-0108', 'jp486-16032023-0108',
       'jp452-25112022-0110', 'jp486-24032023-0108',
       'jp486-22032023-0108', 'jp452-24111022-0109',
       'jp486-15032023-0108', 'jp3120-25052022-0107',
       'jp3120-26052022-0107', 'jp451-28102022-0108',
       'jp486-20032023-0108', 'jp486-06032023-0108',
       'jp486-26032023-0108', 'jp486-17032023-0108',
       'jp451-29102022-0108', 'jp451-30102022-0108',
       'jp486-10032023-0108', 'jp486-05032023-0108',
       'jp3269-29112022-0108', 'mn8578-17122021-0107',
       'jp452-23112022-0108', 'jp1686-26042022-0108']

# Load data
print("Loading data...")
fn = os.path.join(PROJECT_DATA_PATH, "results", "reconstuctionDFAutoPI.csv")
dfAutoPI = pd.read_csv(fn)
print(f"Loaded {len(dfAutoPI)} rows")

# Filter for useable sessions
dfAutoPI = dfAutoPI[dfAutoPI.session.isin(useAble)]
print(f"After filtering for useable sessions: {len(dfAutoPI)} rows")
print(f"Sessions: {dfAutoPI.session.nunique()}")

# Set plot style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 10

def plot_heading_deviation_by_trial(df, condition_name, n_trials=10, output_path=None):
    """
    Plot heading deviation over time for multiple trials.
    
    Parameters:
    -----------
    df : DataFrame
        Filtered dataframe for specific condition
    condition_name : str
        Name of condition for title ('Light' or 'Dark')
    n_trials : int
        Number of trials to plot (default: 10)
    output_path : str
        Path to save figure (optional)
    """
    # Get unique session-trial combinations
    df['session_trial'] = df['session'] + '_' + df['trial'].astype(str)
    unique_trials = df['session_trial'].unique()
    
    print(f"\n{condition_name} trials:")
    print(f"Total unique trials: {len(unique_trials)}")
    
    # Sample trials if more than n_trials
    if len(unique_trials) > n_trials:
        np.random.seed(42)  # For reproducibility
        selected_trials = np.random.choice(unique_trials, n_trials, replace=False)
    else:
        selected_trials = unique_trials
        n_trials = len(unique_trials)
    
    print(f"Plotting {n_trials} trials")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate colors
    colors = plt.cm.tab10(np.linspace(0, 1, n_trials))
    
    # Plot each trial
    for i, trial_id in enumerate(selected_trials):
        trial_data = df[df['session_trial'] == trial_id].copy()
        
        # Calculate time relative to trial start
        trial_data['time_rel'] = trial_data['recTime'] - trial_data['recTime'].min()
        
        # Sort by time
        trial_data = trial_data.sort_values('time_rel')
        
        # Plot
        ax.plot(trial_data['time_rel'], trial_data['mvtDirError'], 
                color=colors[i], alpha=0.6, linewidth=1.5,
                label=f"{trial_id.split('_')[0][:10]}...T{trial_id.split('_')[1]}")
    
    # Format plot
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_xlabel('Time from trial start (s)', fontsize=12)
    ax.set_ylabel('Heading deviation (rad)', fontsize=12)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_yticklabels([r'-$\pi$', r'-$\pi$/2', '0', r'$\pi$/2', r'$\pi$'])
    ax.set_title(f'Heading Deviation Over Time - {condition_name} Trials', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, ncol=1)
    
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
    
    return fig, ax


# Process Light Trials
print("\n" + "="*60)
print("Processing Light Trials")
print("="*60)
dfLight = dfAutoPI[dfAutoPI.condition == 'all_light'].copy()
print(f"Light trials: {len(dfLight)} rows, {dfLight.session.nunique()} sessions")

fig_light, ax_light = plot_heading_deviation_by_trial(
    dfLight, 
    'Light', 
    n_trials=10,
    output_path=os.path.join(PROJECT_DATA_PATH, 'results', 'heading_deviation_light_trials.png')
)

# Process Dark Trials
print("\n" + "="*60)
print("Processing Dark Trials")
print("="*60)
dfDark = dfAutoPI[dfAutoPI.condition == 'all_dark'].copy()
print(f"Dark trials: {len(dfDark)} rows, {dfDark.session.nunique()} sessions")

fig_dark, ax_dark = plot_heading_deviation_by_trial(
    dfDark, 
    'Dark', 
    n_trials=10,
    output_path=os.path.join(PROJECT_DATA_PATH, 'results', 'heading_deviation_dark_trials.png')
)

print("\n" + "="*60)
print("Figures generated successfully!")
print("="*60)

# Show plots
plt.show()
