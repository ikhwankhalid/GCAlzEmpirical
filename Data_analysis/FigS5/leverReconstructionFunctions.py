#Functions I use to make the plots and also preprocess the data
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.stats import poisson
from scipy.interpolate import interp1d
from tqdm import tqdm
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy import ndimage
from math import remainder, tau
import math
import matplotlib.pyplot as plt


#Functions to get grid coordinates
def get_v(df,predicted=True):
    
    if predicted:
        v0 = np.arctan2(df['v0_sin_smooth'],df['v0_cos_smooth'])  
        v1 = np.arctan2(df['v1_sin_smooth'],df['v1_cos_smooth'])
    else: 
        v0 = np.arctan2(df['lv0_sin_smooth'],df['lv0_cos_smooth'])
        v1 = np.arctan2(df['lv1_sin_smooth'],df['lv1_cos_smooth'])
    
    return v0, v1


def get_double_v(df,predicted=True):
    
    if predicted:
        v0 = list(np.arctan2(df['v0_sin_smooth'],df['v0_cos_smooth']))
        v1 = list(np.arctan2(df['v1_sin_smooth'],df['v1_cos_smooth']))
    else: 
        v0 = list(np.arctan2(df['lv0_sin_smooth'],df['lv0_cos_smooth']))
        v1 = list(np.arctan2(df['lv1_sin_smooth'],df['lv1_cos_smooth']))
    
    return [v+2*np.pi for v in v0]+[v+2*np.pi for v in v0]+v0+v0,\
           [v+2*np.pi for v in v1]+v1+[v+2*np.pi for v in v1]+v1

    
remainder_fn = lambda x: math.remainder(x, tau)
    
def get_lever_v(df,predicted=True):
    
    if predicted:
        v0 = np.arctan2(df['v0_sin_smooth'],df['v0_cos_smooth']) + df['vectorToLever_v0']
        v1 = np.arctan2(df['v1_sin_smooth'],df['v1_cos_smooth']) + df['vectorToLever_v1']
    else: 
        v0 = np.arctan2(df['lv0_sin_smooth'],df['lv0_cos_smooth']) + df['vectorToLever_v0']
        v1 = np.arctan2(df['lv1_sin_smooth'],df['lv1_cos_smooth']) + df['vectorToLever_v1']
    
    v0 = np.array(list(map(remainder_fn, v0)))
    v1 = np.array(list(map(remainder_fn, v1)))
    
    return v0, v1

def get_double_lever_v(df,predicted=True):
    
    if predicted:
        v0 = list(np.arctan2(df['v0_sin_smooth'],df['v0_cos_smooth']) + df['vectorToLever_v0'])
        v1 = list(np.arctan2(df['v1_sin_smooth'],df['v1_cos_smooth']) + df['vectorToLever_v1'])
    else: 
        v0 = list(np.arctan2(df['lv0_sin_smooth'],df['lv0_cos_smooth']) + df['vectorToLever_v0'])
        v1 = list(np.arctan2(df['lv1_sin_smooth'],df['lv1_cos_smooth']) + df['vectorToLever_v1'])
        
    v0 = list(map(remainder_fn, v0))
    v1 = list(map(remainder_fn, v1))
    
    return [v+2*np.pi for v in v0]+[v+2*np.pi for v in v0]+v0+v0,\
           [v+2*np.pi for v in v1]+v1+[v+2*np.pi for v in v1]+v1

#Plotting Function
def plot_heatmap(ax,va,vb,x_label,y_label,title):
    
    smoothing = 2
    heatmap, xedges, yedges = np.histogram2d(va,vb,bins=40)
    heatmap = ndimage.gaussian_filter(heatmap,sigma=smoothing,mode="wrap")
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    ax.imshow(heatmap.T,origin="lower",extent=[-np.pi,np.pi,-np.pi,np.pi])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(False)
    ax.set_title(f'{title}')
    

    ax.set_xticks(ticks=[-np.pi, 0, np.pi])
    ax.set_xticklabels([r'-$\pi$', "0", "$\pi$"])
    ax.set_yticks(ticks=[-np.pi, 0, np.pi])
    ax.set_yticklabels([r'-$\pi$', "0", "$\pi$"])
    
def plot_double_heatmap(ax,va,vb,x_label,y_label,title):
    
    smoothing = 2
    heatmap, xedges, yedges = np.histogram2d(va,vb,bins=40)
    heatmap = ndimage.gaussian_filter(heatmap,sigma=smoothing,mode="wrap")
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    ax.imshow(heatmap.T,origin="lower",extent=[-np.pi,3*np.pi,-np.pi,3*np.pi])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(False)
    ax.set_title(f'Enlarged {title}')
    

    ax.set_xticks(ticks=[-np.pi, 0, 3*np.pi])
    ax.set_xticklabels([r'-$\pi$', "0", "3$\pi$"])
    ax.set_yticks(ticks=[-np.pi, 0, 3*np.pi])
    ax.set_yticklabels([r'-$\pi$', "0", "3$\pi$"])
    
    
def plot_different_coordinates(ax1,predicted=True,enlarge=False,session = 'jp452-25112022-0110',light = 'atLever_dark'):
    minSpeed = 10
    TestdfAutoPI = dfAutoPI[dfAutoPI.session == session].copy()
    S = TestdfAutoPI[(TestdfAutoPI.condition == light)& (TestdfAutoPI.speed > minSpeed)].copy()
    
    
    S = S.dropna(subset = ['lv0_cos_smooth', 'lv0_sin_smooth','lv1_cos_smooth','lv1_sin_smooth']).reset_index(drop=True)
    
    v0, v1 = get_v(S,predicted)
    dv0,dv1 = get_double_v(S,predicted)

    
    if predicted:
        t = 'Predicted'
    else:
        t = 'Actual'
    
    if enlarge == False:
        plot_heatmap(ax1,v0,v1,'v0','v1',t)
    else:
        plot_double_heatmap(ax1,dv0,dv1,'v0','v1',t)
    
    return v0, v1
        


def plot_different_trials(ax1,predicted=True,enlarge=False,session = 'jp452-25112022-0110',light = 'atLever_dark',trials=0):
    minSpeed = 10
    TestdfAutoPI = dfAutoPI[dfAutoPI.session == session].copy()
    S = TestdfAutoPI[(TestdfAutoPI.condition == light)& (TestdfAutoPI.speed > minSpeed) ].copy()
    S = S[S.trial==trials].copy()
    
    
    S = S.dropna(subset = ['lv0_cos_smooth', 'lv0_sin_smooth','lv1_cos_smooth','lv1_sin_smooth']).reset_index(drop=True)
    
    v0, v1 = get_v(S,predicted)
    dv0,dv1 = get_double_v(S,predicted)

    
    if predicted:
        t = 'Predicted'
    else:
        t = 'Actual'
    
    if enlarge == False:
        plot_heatmap(ax1,v0,v1,'v0','v1',t)
    else:
        plot_double_heatmap(ax1,dv0,dv1,'v0','v1',t)
        
        
def plot_openfield_coordinates(ax1,predicted=True,enlarge=False,session = 'jp3269-28112022-0108'):
    minSpeed = 10
    TestdfAutoPI = dfOF[dfOF.session == session].copy()
    S = TestdfAutoPI[(TestdfAutoPI.speed > minSpeed) ].copy()
    
    
    S = S.dropna(subset = ['lv0_cos_smooth', 'lv0_sin_smooth','lv1_cos_smooth','lv1_sin_smooth']).reset_index(drop=True)
    
    v0, v1 = get_v(S,predicted)
    dv0,dv1 = get_double_v(S,predicted)

    
    if predicted:
        t = 'Predicted'
    else:
        t = 'Actual'
    
    if enlarge == False:
        plot_heatmap(ax1,v0,v1,'v0','v1',t)
    else:
        plot_double_heatmap(ax1,dv0,dv1,'v0','v1',t)
        
    return v0, v1

arenaBorderColor = "#BAD7E9"

def draw_circle(ax, r=44, c=arenaBorderColor, center=(0, 0), lw=2, ls="solid"):

    # Define the center coordinates and radius of the circle
    center = center
    radius = r

    # Create the circle patch
    circle = plt.Circle(
        center, radius, edgecolor=c, facecolor="none", lw=lw, linestyle=ls, alpha=1
    )

    # Add the circle patch to the axes
    ax.add_patch(circle)

    # Set the aspect ratio to 'equal' to ensure the circle appears as a circle
    ax.set_aspect("equal")

    # Set the x and y axis limits
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)