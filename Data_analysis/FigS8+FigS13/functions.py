import numpy as np
import torch
import cv2
from scipy.stats import wilcoxon, pearsonr
import datetime
import os.path
import shutil
import pickle
from spikeA.Neuron import Simulated_place_cell, Simulated_grid_cell
from scipy import stats
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
# from generic_plot_functions import plotMap
from tqdm import tqdm

from scipy.stats import vonmises
from scipy.optimize import minimize

from spikeA.Session import Kilosort_session
from spikeA.Animal_pose import Animal_pose
from spikeA.Spike_train_loader import Spike_train_loader
from spikeA.Cell_group import Cell_group

import pandas as pd

from scipy import ndimage




def fit_grid_parameter_from_grid_cell_activity(n,ap,apSim,interval,cm_per_bin = 3,xy_range=np.array([[-50,-90],[50,60]]),n_epochs=5000):
    """
    Function that finds the best grid cell model parameters (period, orientation, peak rate, offset) that predict the firing rate of the neuron
    
    The model takes x,y position as input and try to predict the firing rate of the neuron.
    
    We start with an estimate from firing rate map (offset) and spatial autocorrelation (orientation, spacing), then fit some grid cell models with gradient descent.
    
    
    """
    rowSize,colSize= figurePanelDefaultSize()
    
    ap.set_intervals(interval)
    n.spike_train.set_intervals(interval)
    n.spatial_properties.set_intervals(interval)
    n.spatial_properties.firing_rate_map_2d(cm_per_bin =cm_per_bin, smoothing_sigma_cm = 5, smoothing=True,xy_range=xy_range)
    n.spatial_properties.spatial_autocorrelation_map_2d(min_n_for_correlation=50,invalid_to_nan=True)
    n.spatial_properties.grid_score()
    n.spike_train.instantaneous_firing_rate(bin_size_sec = 0.02,sigma=1,outside_interval_solution="remove")

    gcSpacing = n.spatial_properties.grid_info()[0]
    gcPeakLocation = n.spatial_properties.firing_rate_map_peak_location()

    # get a template spatial autocorrelation with orientation set at 0
    grid_param = {}
    period = gcSpacing * np.cos(np.pi/6)
    grid_param["period"] = np.array([period,period,period])
    grid_param["offset"] = gcPeakLocation[1]
    grid_param["peak_rate"] = 25
    grid_param["orientation"] = np.array([0,np.pi/3,np.pi/3*2]) # 30 degree orientation means that the field to the right will be at 1,0
    apSim.set_intervals(interval)

    print(grid_param)
    sgc = Simulated_grid_cell(name="pc1",
                              offset=grid_param["offset"],
                              orientation=grid_param["orientation"],
                              period=grid_param["period"],
                              peak_rate=grid_param["peak_rate"],
                              ap=apSim)

    apSim.set_intervals(interval)
    sgc.spike_train.set_intervals(interval)
    sgc.spatial_properties.firing_rate_map_2d(cm_per_bin =cm_per_bin, smoothing_sigma_cm = 5, smoothing=True,xy_range=xy_range)
    sgc.spatial_properties.spatial_autocorrelation_map_2d(min_n_for_correlation=50,invalid_to_nan=True)

    refAuto = n.spatial_properties.spatial_autocorrelation_map.copy()
    invalidate_surrounding(myMap=refAuto,cm_per_bin=cm_per_bin,valid_radius_cm=gcSpacing+gcSpacing*0.4)
    refAutoSgc = sgc.spatial_properties.spatial_autocorrelation_map.copy()
    invalidate_surrounding(myMap=refAutoSgc,cm_per_bin=cm_per_bin,valid_radius_cm=gcSpacing+gcSpacing*0.4)
    autoStackReal = np.expand_dims(refAuto,0)
    autoStackSim = np.expand_dims(refAutoSgc,0)
    
    rot,cor = rotation_correlations(autoStackReal,autoStackSim,minRotation=0,maxRotation=np.pi,nRotations=180)
    peak_indices,_ = find_peaks(cor,height=0.3,distance=10)
    deltas = rot[peak_indices]
    
    print("Results after rotating the spatial autocorrelation")
    print("First axis at ",deltas[0])
    ncols=1
    nrows=1
    fig = plt.figure(figsize=(ncols*4, nrows*rowSize), constrained_layout=True) # create a figure
    mainSpec = fig.add_gridspec(ncols=ncols, nrows=nrows)
    ax = fig.add_subplot(mainSpec[0])
    ax.plot(rot,cor)
    ax.scatter(rot[peak_indices],cor[peak_indices],color="red",s=10)
    ax.set_xlabel("Rotation")
    ax.set_ylabel("r value")
    plt.show()
    
    
    if len(deltas) > 4:
        raise ValueError("Expect less than 5 peaks while rotating spatial autocorrelation but got {}".format(len(deltas)))
    
    if len(deltas) < 3:
        raise ValueError("Expect at least 5 peaks while rotating spatial autocorrelation but got {}".format(len(deltas)))
    grid_param["orientation"] = np.array(deltas[0:3]) # take the first 3
    
   
    # get a simulated grid cells with our initial grid parameters
    sgc = Simulated_grid_cell(name="pc1",
                              offset=grid_param["offset"],
                              orientation=grid_param["orientation"],
                              period=grid_param["period"],
                              peak_rate=grid_param["peak_rate"],
                              ap=apSim)

    apSim.set_intervals(interval)
    sgc.spike_train.set_intervals(interval)
    sgc.spatial_properties.firing_rate_map_2d(cm_per_bin =cm_per_bin, smoothing_sigma_cm = 5, smoothing=True,xy_range=xy_range)
    
    print("Visualize real and simulated grid pattern before fitting")
    ncols=2
    nrows=1
    fig = plt.figure(figsize=(ncols*colSize, nrows*rowSize), constrained_layout=True) # create a figure
    mainSpec = fig.add_gridspec(ncols=ncols, nrows=nrows)
    ax = fig.add_subplot(mainSpec[0])
    plotMap(ax,n.spatial_properties.firing_rate_map,
            title="Data - {:.2f} Hz".format(np.nanmax(n.spatial_properties.firing_rate_map)),
            titleY=0.95,titleFontSize=9,transpose=True,cmap="jet",vmin=0)
    ax = fig.add_subplot(mainSpec[1])
    plotMap(ax,sgc.spatial_properties.firing_rate_map,
            title="Sim - {:.2f} Hz".format(np.nanmax(sgc.spatial_properties.firing_rate_map)),
            titleY=0.95,titleFontSize=9,transpose=True,cmap="jet",vmin=0)
    plt.show()
    
    # get the data that will be used for modelling
    ap.set_intervals(interval)
    
    # trick to get aligned ifr and pose data
    modInterval = interval.copy()
    modInterval[0,0] = ap.pose[0,0]
    modInterval[0,1] = ap.pose[-1,0]+0.00000001
    
    n.spike_train.set_intervals(modInterval)
    n.spike_train.instantaneous_firing_rate(bin_size_sec = 0.02,sigma=1,shift_start_time = -0.0099999999,
                                            outside_interval_solution="remove")

    poseLong = ap.pose[:,1:3]
    rateLong = n.spike_train.ifr[0].copy()
        
    # remove np.nan
    keepIndices = ~np.any(np.isnan(poseLong),1)
    rate = rateLong[keepIndices]
    pose = poseLong[keepIndices]
    #print(rate.shape, pose.shape)
    
    # transform to tensors
    tpose = torch.tensor(pose,dtype=torch.float32)
    trate = torch.tensor(np.expand_dims(rate,1),dtype=torch.float32)
    print("Shape of tensors used for training:",tpose.shape,trate.shape)
    
    
    # get a rigid grid model
    rgcModel = RigidGridCellModel(period=grid_param["period"][0], 
                              peak_rate=grid_param["peak_rate"],
                              orientation=grid_param["orientation"][0],
                              offset=grid_param["offset"])
    
    grid_param_model_start = rgcModel.modelParamToGridParam()
    grid_param_model_start
    print("Fitting rigid grid cell model")
    loss_rigid = training_loop_grid_parameters(n_epochs = n_epochs,
                          model=rgcModel,
                          optimizer=torch.optim.Adam(rgcModel.parameters(),lr=0.01),
                          loss_fn = torch.nn.MSELoss(),
                          X = tpose,
                          y = trate,
                         verbose=False)
    loss_rigid = loss_rigid.detach().numpy()
    print("Loss after rigid model fitting:",loss_rigid)
    
    grid_param_model_rigid = rgcModel.modelParamToGridParam()
    sgcRigid = Simulated_grid_cell(name="pc1",
                          offset=grid_param_model_rigid["offset"],
                          orientation=grid_param_model_rigid["orientation"],
                          period=grid_param_model_rigid["period"],
                          peak_rate=grid_param_model_rigid["peak_rate"],
                          ap=apSim)
    
    sgcRigid.spatial_properties.firing_rate_map_2d(cm_per_bin=cm_per_bin,smoothing_sigma_cm=3, xy_range=xy_range)
    rStart = sgcRigid.spatial_properties.map_pearson_correlation(map1=n.spatial_properties.firing_rate_map,
                                            map2=sgc.spatial_properties.firing_rate_map)
    rRigid = sgcRigid.spatial_properties.map_pearson_correlation(map1=n.spatial_properties.firing_rate_map,
                                            map2=sgcRigid.spatial_properties.firing_rate_map)
    print("Improvement of firing rate maps correlation after rigid fitting: from {:.3f} to {:.3f}".format(rStart, rRigid))
    
    gcModel = GridCellModel(period=grid_param_model_rigid["period"], 
                        peak_rate=grid_param_model_rigid["peak_rate"], 
                        orientation=grid_param_model_rigid["orientation"],
                        offset=grid_param_model_rigid["offset"])
    print("Fitting more flexible grid cell model")
    loss_flexible = training_loop_grid_parameters(n_epochs = n_epochs,
                          model=gcModel,
                          optimizer=torch.optim.Adam(gcModel.parameters(),lr=0.01),
                          loss_fn = torch.nn.MSELoss(),
                          X = tpose,
                          y = trate,
                         verbose=False)
    loss_flexible = loss_flexible.detach().numpy()
    print("Loss after flexible model fitting:",loss_flexible)
    grid_param_model_flexible = gcModel.modelParamToGridParam()
    
    sgcFlexible = Simulated_grid_cell(name="pc1",
                          offset=grid_param_model_flexible["offset"],
                          orientation=grid_param_model_flexible["orientation"],
                          period=grid_param_model_flexible["period"],
                          peak_rate=grid_param_model_flexible["peak_rate"],
                          ap=apSim)
    sgcFlexible.spatial_properties.firing_rate_map_2d(cm_per_bin=cm_per_bin,smoothing_sigma_cm=3, xy_range=xy_range)
    rFlexible = sgc.spatial_properties.map_pearson_correlation(map1=n.spatial_properties.firing_rate_map,
                                                map2=sgcFlexible.spatial_properties.firing_rate_map)
    print("Improvement of firing rate maps correlation after flexible fitting: from {:.3f} to {:.3f}".format(rRigid,rFlexible))
    
    print("Comparison of firing rate maps after fitting different models")
    
    ncols=4
    nrows=1
    fig = plt.figure(figsize=(ncols*colSize, nrows*rowSize), constrained_layout=True) # create a figure
    mainSpec = fig.add_gridspec(ncols=ncols, nrows=1)

    ax = fig.add_subplot(mainSpec[0])
    plotMap(ax,n.spatial_properties.firing_rate_map)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Original",fontsize=9)

    ax = fig.add_subplot(mainSpec[1])
    plotMap(ax,sgc.spatial_properties.firing_rate_map)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Simulated not fitted\n {:.3f}".format(rStart),fontsize=9)

    ax = fig.add_subplot(mainSpec[2])
    plotMap(ax,sgcRigid.spatial_properties.firing_rate_map)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Simulated rigid fit\n {:.3f}".format(rRigid),fontsize=9)

    ax = fig.add_subplot(mainSpec[3])
    plotMap(ax,sgcFlexible.spatial_properties.firing_rate_map)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Simulated flexible fit\n {:.3f}".format(rFlexible),fontsize=9)

    plt.show()
    
    res = {"name":n.name,
           "grid_param_initial":grid_param,
           "grid_param_model_rigid": grid_param_model_rigid,
           "loss_rigid": loss_rigid,
           "r_rigid": rRigid,
           "grid_param_model_flexible": grid_param_model_flexible,
           "loss_flexible": loss_flexible,
           "r_flexible": rFlexible}
    return res
       
















def figurePanelDefaultSize():
    """
    Use to keep the size of panels similar across figures
    """
    return (1.8,1.8)



def invalidate_surrounding(myMap,cm_per_bin=3, valid_radius_cm=50):
    xs,ys = np.meshgrid(np.arange(0,myMap.shape[0]),np.arange(0,myMap.shape[1]))
    midPoint=(myMap.shape[0]/2,myMap.shape[1]/2)
    distance = np.sqrt((xs.T-midPoint[0])**2 + (ys.T-midPoint[1])**2) * cm_per_bin
    myMap[distance>valid_radius_cm]=np.nan
    
    
def rotation_correlations(ccStack1,ccStack2,minRotation=-np.pi,maxRotation=np.pi,nRotations=72):
    """
    We rotate the maps of ccStack2 and perform a correlation coefficient between individual cc.
    We create a 2D array of correlation coefficients (rotation x cell_pairs)
    Then we get the mean coefficient at each rotation
    """
    rotations = np.linspace(minRotation,maxRotation,nRotations)
    rotatedStack = np.empty_like(ccStack2)
    corrValues = np.empty((rotations.shape[0],ccStack2.shape[0]))

    for i,r in enumerate(rotations):
        for j in range(ccStack2.shape[0]): # rotate individual maps
            rotatedStack[j,:,:] = rotate_map(ccStack2[j,:,:],rotation_radian=r)
            corrValues[i,j] = map_cor(ccStack1[j,:,:],rotatedStack[j,:,:])

    peaks = np.mean(corrValues,axis=1)
    return rotations, peaks

def map_cor(a,b):
    """
    Correlation coefficient between two firing rate maps
    
    Arguments:
    a: 2D np.array (map1)
    b: 2D np.array (map2)
    
    Returns:
    Pearson correlation coefficient between a and b
    """
    a = a.flatten()
    b = b.flatten()
    indices1 = np.logical_and(~np.isnan(a), ~np.isnan(b))
    indices2 = np.logical_and(~np.isinf(a), ~np.isinf(b))
    indices = np.logical_and(indices1,indices2)
    if np.sum(indices)<2:
        return np.nan
    r,p = pearsonr(a[indices],b[indices])
    return r

def rotate_map(a,rotation_radian=np.pi/6):
    """
    Rotate the values in a map around the center of the map
    
    Arguments:
    a: 2D Numpy array
    rotation_radian: angle of the rotation in radian, positive is anti-clockwise
    
    Return:
    2D Numpy array with values rotated
    """

    (h, w) = a.shape
    (cX, cY) = (w // 2, h // 2) # center of the rotation
    degree = rotation_radian/(2*np.pi)*360

    # rotate by degree°, same scale
    M = cv2.getRotationMatrix2D(center = (cX, cY), angle = degree, scale = 1.0)
    a_rotated = cv2.warpAffine(a, M, (w, h), borderValue = np.nan)
    return a_rotated

def load_session_files_for_modelling(sessionName,cells,sSessions):
    
    ## load data from the session
    sSes =[ses for ses in sSessions if ses.name == sessionName][0]
    pose_file_extension = ".pose.npy"
    sSes.load_parameters_from_files() 
    ap = Animal_pose(sSes)
    ap.pose_file_extension = pose_file_extension # This means that the ap will always load from this extension
    ap.load_pose_from_file()
    apSim = Animal_pose(sSes)
    apSim.pose_file_extension = pose_file_extension # This means that the ap will always load from this extension
    apSim.load_pose_from_file()
    stl = Spike_train_loader()
    stl.load_spike_train_kilosort(sSes)
    cg = Cell_group(stl,ap)
    
    # create a list of grid cells (spikeA.Neuron)
    gcIds = cells[(cells.session==sessionName)&(cells.gridCell_FIRST==True)].cluId
    gcIds = [gcId.split("_")[1] for gcId in gcIds]
    cg.gc_list = [ n for n in cg.neuron_list if n.name in gcIds ]
    #print(sessionName, ",",len(cg.gc_list),"grid cells")
    
    return sSes, ap, apSim, cg



def visualize_grid_cell_parameters(sessionName,cells,sSessions):
    # load session files
    print(sessionName)
    sSes, ap, apSim, cg = load_session_files_for_modelling(sessionName,cells,sSessions)
    
    # load grid cell parameters
    fn = sSes.fileBase+".grid_cell_parameters.pkl"
    #print("Loading:",fn)
    with open(fn, 'rb') as fp: 
        params = pickle.load(fp)
    oriRigid = np.stack([p["grid_param_model_rigid"]["orientation"] for p in params])
    oriFlexible = np.stack([p["grid_param_model_flexible"]["orientation"] for p in params])
    periodRigid = np.stack([p["grid_param_model_rigid"]["period"] for p in params])
    periodFlexible = np.stack([p["grid_param_model_flexible"]["period"] for p in params])

    # make the figure
    rowSize,colSize= figurePanelDefaultSize()
    ncols=2
    nrows=2
    fig = plt.figure(figsize=(ncols*colSize, nrows*rowSize), constrained_layout=True) # create a figure
    mainSpec = fig.add_gridspec(ncols=ncols, nrows=nrows)

    ax = fig.add_subplot(mainSpec[0,0])
    ax.hist(oriRigid[:,0],bins=np.linspace(-np.pi,np.pi,72),label="c0")
    ax.hist(oriRigid[:,1],bins=np.linspace(-np.pi,np.pi,72),label="c1")
    ax.hist(oriRigid[:,2],bins=np.linspace(-np.pi,np.pi,72),label="c2")
    ax.set_xlim(-np.pi,np.pi)
    ax.set_xlabel("Orientation")
    ax.set_title("Orientation (rigid)",fontsize=9)
    ax.legend()

    ax = fig.add_subplot(mainSpec[0,1])
    ax.hist(oriFlexible[:,0],bins=np.linspace(-np.pi,np.pi,72),label="c0")
    ax.hist(oriFlexible[:,1],bins=np.linspace(-np.pi,np.pi,72),label="c1")
    ax.hist(oriFlexible[:,2],bins=np.linspace(-np.pi,np.pi,72),label="c2")
    ax.set_xlim(-np.pi,np.pi)
    ax.set_xlabel("Orientation")
    ax.set_title("Orientation (flexible)",fontsize=9)
    ax.legend()


    ax = fig.add_subplot(mainSpec[1,0])
    ax.hist(periodRigid[:,0],bins=np.linspace(25,50,25),label="c0-c1-c2")
    ax.set_xlim(25,50)
    ax.set_xlabel("Period")
    ax.set_title("Period (rigid)",fontsize=9)
    ax.legend()

    ax = fig.add_subplot(mainSpec[1,1])
    ax.hist(periodFlexible[:,0],bins=np.linspace(25,50,25),label="c0",alpha=0.5)
    ax.hist(periodFlexible[:,1],bins=np.linspace(25,50,25),label="c1",alpha=0.5)
    ax.hist(periodFlexible[:,2],bins=np.linspace(25,50,25),label="c2",alpha=0.5)
    #ax.set_xlim(-np.pi,np.pi)
    ax.set_xlabel("Period")
    ax.set_title("Period (flexible)",fontsize=9)
    ax.legend()
    plt.show()





class RigidGridCellModel(torch.nn.Module):
    """
    Model with one spacing, one orientation and one peak rate
    """
    def __init__(self,period,orientation,peak_rate,offset):
        super(RigidGridCellModel, self).__init__()

        period = torch.tensor([period],requires_grad =True,dtype=torch.float32) # model parameters
        offset = torch.tensor(offset,requires_grad =True,dtype=torch.float32) # model parameters
        peak_rate = torch.tensor([peak_rate],requires_grad =True,dtype=torch.float32) # model parameters
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        

        self.period = torch.nn.Parameter(period)
        self.offset = torch.nn.Parameter(offset)
        self.peak_rate = torch.nn.Parameter(peak_rate)
        
        # orientation
        self.ori_scalling= 0.01 # to make the gradient for orientation similar to other parameters
        ori = torch.tensor([orientation/self.ori_scalling], requires_grad=True,dtype=torch.float32) # start with 60 degree orientation
        self.ori = torch.nn.Parameter(ori)
       
        
        ## matrix to get the cos and sin component for our 2,1 projection matrix
        self.myMatCos = torch.tensor([[1],[0]],dtype=torch.float32)
        self.myMatSin = torch.tensor([[0],[1]],dtype=torch.float32)
       
        
    def length_to_angle(self,x,period):
        xr = x/period*np.pi*2
        return (torch.atan2(torch.sin(xr), torch.cos(xr)))

    def forward(self, X):
       
        # matrix to project onto the x axis and keep only the x coordinate
        self.sori = self.ori * self.ori_scalling
       
        self.Rx0 = self.myMatCos @ torch.cos(-self.sori[0].reshape(1,1))+ self.myMatSin @ -torch.sin(-self.sori[0].reshape(1,1)) # minus sign because we want to rotate the inverse of the angle to bring it back to 1,0
        self.Rx1 = self.myMatCos @ torch.cos(- (self.sori[0].reshape(1,1)+self.pi/3)) + self.myMatSin @ -torch.sin(-(self.sori[0].reshape(1,1)+self.pi/3))
        self.Rx2 = self.myMatCos @ torch.cos(- (self.sori[0].reshape(1,1)+self.pi/3*2)) + self.myMatSin @ -torch.sin(-(self.sori[0].reshape(1,1)+self.pi/3*2))
         
        
        # distance in cm along each axis
        d0 = X @ self.Rx0
        d1 = X @ self.Rx1
        d2 = X @ self.Rx2
        
        c0 = self.length_to_angle(d0, self.period[0])
        c1 = self.length_to_angle(d1, self.period[0]) 
        c2 = self.length_to_angle(d2, self.period[0])

        # deal with the offset, project on each vector 
        d0 = self.offset @ self.Rx0
        d1 = self.offset @ self.Rx1
        d2 = self.offset @ self.Rx2

        # offset as angle
        a0 = self.length_to_angle(d0, self.period[0])
        a1 = self.length_to_angle(d1, self.period[0])
        a2 = self.length_to_angle(d2, self.period[0])
      
        rateC0 = torch.cos(c0-a0)
        rateC1 = torch.cos(c1-a1)
        rateC2 = torch.cos(c2-a2)

        rate = (rateC0+rateC1+rateC2+1.5)/4.5*self.peak_rate
        return rate

    def modelParamToGridParam(self):
        """
        Return the grid cell parameters in a dictionary
        """
        myIter = iter(self.parameters())
        pred_grid_param = {}
        period = next(myIter).detach().numpy()
        pred_grid_param["period"] = np.array([period[0],period[0],period[0]])
        pred_grid_param["offset"] = next(myIter).detach().numpy()
        pred_grid_param["peak_rate"] = next(myIter).detach().numpy()
        ori = next(myIter).detach().numpy() * self.ori_scalling
        pred_grid_param["orientation"] = np.array([ori[0],ori[0]+np.pi/3,ori[0]+np.pi/3*2])
        return pred_grid_param

class GridCellModel(torch.nn.Module):
    """
    Model with 3 period, 3 orientation and 1 peak rate.
    """
    def __init__(self,period,orientation,peak_rate,offset):
        super(GridCellModel, self).__init__()

        period = torch.tensor(period,requires_grad =True,dtype=torch.float32) # model parameters
        offset = torch.tensor(offset,requires_grad =True,dtype=torch.float32) # model parameters
        peak_rate = torch.tensor(peak_rate,requires_grad =True,dtype=torch.float32) # model parameters
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        

        self.period = torch.nn.Parameter(period)
        self.offset = torch.nn.Parameter(offset)
        self.peak_rate = torch.nn.Parameter(peak_rate)
        
        # orientation
        self.ori_scalling= 0.01 # to make the gradient for orientation similar to other parameters
        ori = torch.tensor(orientation/self.ori_scalling, requires_grad=True,dtype=torch.float32) # start with 60 degree orientation
        self.ori = torch.nn.Parameter(ori)
       
        
        ## matrix to get the cos and sin component for our 2,1 projection matrix
        self.myMatCos = torch.tensor([[1],[0]],dtype=torch.float32)
        self.myMatSin = torch.tensor([[0],[1]],dtype=torch.float32)
       
        
    def length_to_angle(self,x,period):
        xr = x/period*np.pi*2
        return (torch.atan2(torch.sin(xr), torch.cos(xr)))

    def forward(self, X):
       
        # matrix to project onto the x axis and keep only the x coordinate
        self.sori = self.ori * self.ori_scalling
       
        self.Rx0 = self.myMatCos @ torch.cos(-self.sori[0].reshape(1,1))+ self.myMatSin @ -torch.sin(-self.sori[0].reshape(1,1)) # minus sign because we want to rotate the inverse of the angle to bring it back to 1,0
        self.Rx1 = self.myMatCos @ torch.cos(-self.sori[1].reshape(1,1)) + self.myMatSin @ -torch.sin(-self.sori[1].reshape(1,1))
        self.Rx2 = self.myMatCos @ torch.cos(-self.sori[2].reshape(1,1)) + self.myMatSin @ -torch.sin(-self.sori[2].reshape(1,1))
         
        
        # distance in cm along each axis
        d0 = X @ self.Rx0
        d1 = X @ self.Rx1
        d2 = X @ self.Rx2
        
        c0 = self.length_to_angle(d0, self.period[0])
        c1 = self.length_to_angle(d1, self.period[1]) 
        c2 = self.length_to_angle(d2, self.period[2])

        # deal with the offset, project on each vector 
        d0 = self.offset @ self.Rx0
        d1 = self.offset @ self.Rx1
        d2 = self.offset @ self.Rx2

        # offset as angle
        a0 = self.length_to_angle(d0, self.period[0])
        a1 = self.length_to_angle(d1, self.period[1])
        a2 = self.length_to_angle(d2, self.period[2])
      
        rateC0 = torch.cos(c0-a0)
        rateC1 = torch.cos(c1-a1)
        rateC2 = torch.cos(c2-a2)

        rate = (rateC0+rateC1+rateC2+1.5)/4.5*self.peak_rate
        return rate

    def modelParamToGridParam(self):
        """
        Return the model parameters as a dictionary
        """
        myIter = iter(self.parameters())
        pred_grid_param = {}
        pred_grid_param["period"] = next(myIter).detach().numpy()
        pred_grid_param["offset"] = next(myIter).detach().numpy()
        pred_grid_param["peak_rate"] = next(myIter).detach().numpy()

        ori = next(myIter).detach().numpy() * self.ori_scalling
        orientation = np.array([ori[0],ori[1],ori[2]])

        pred_grid_param["orientation"] = orientation
        return pred_grid_param

## training loop that will modify our parameters to minimize the loss function (MSE)
def training_loop_grid_parameters(n_epochs, model, optimizer, loss_fn, X,y,verbose=True):
    
    for epoch in range (n_epochs):
        #for X,y in data_loader:
        optimizer.zero_grad()
        yhat = model(X)
        loss = loss_fn(yhat,y)
        loss.backward()
        optimizer.step()

        if epoch % 1000 ==0:
            if verbose:
                print("Epoch: {}, Loss: {}".format(epoch,loss))
                print("Parameters")
                pred_grid_param = model.modelParamToGridParam()
                print(pred_grid_param)
                for param in model.parameters():
                    print(param)
                print("Gradients")
                for param in model.parameters():
                    print(param.grad)
                print("")

        if loss < 0.0001:
            if verbose:
                print("Final loss:", loss)
            return loss
    return loss



def get_ifr_for_model(sessionName,cells,sSessions,intervals,verbose=False):
    """
    Get ifr that matches the ap.pose samples
    
    """
    
    # load session files
    sSes, ap, apSim, cg = load_session_files_for_modelling(sessionName,cells,sSessions)
    
    if verbose:
        print(intervals)
    
    ap.set_intervals(intervals)
    
    # trick to get aligned ifr and pose data
    modIntervals = intervals.copy()
    modIntervals[0,0] = ap.pose[0,0]
    modIntervals[0,1] = ap.pose[-1,0]+0.0000001
    
    #print("modIntervals:", modIntervals)
    for n in cg.gc_list:
        n.spike_train.set_intervals(modIntervals)
        n.spike_train.instantaneous_firing_rate(bin_size_sec = 0.02,sigma=1,shift_start_time = -0.0099999999,
                                                outside_interval_solution="remove")
    
    poseLong = ap.pose[:,1:3]
    keepIndices = ~np.any(np.isnan(poseLong),1)
    
    if verbose:
        print("Start ap time:", ap.pose[0,0], ", start ifr time:", n.spike_train.ifr[2][0])
        print("End ap time:", ap.pose[-1,0], ", end ifr time:", n.spike_train.ifr[2][-1])
    
    
    
    if keepIndices.shape[0] != n.spike_train.ifr[0].shape[0]:
        raise IndexError("keepIndices.shape[0]: {}, n.spike_train.ifr[0].shape[0]: {}".format(keepIndices.shape[0], 
                                                                                              n.spike_train.ifr[0].shape[0]))
    
    
    ifr = np.stack([n.spike_train.ifr[0][keepIndices] for n in cg.gc_list])
    return ifr  
    



def train_rnn_on_first_open_field(sessionName,cells,sSessions):
    """
    Function to train our rnn using the first random foraging trial.
    
    We used parameters for training that we got from simulation. See https://github.com/kevin-allen/lstm_spike_to_position/blob/main/03_learning_simulated_data.ipynb
    """
    
    sSes, ap, apSim, cg = load_session_files_for_modelling(sessionName,cells,sSessions)
    print(sSes.desen)
    
    ## get training and test intervals
    trainingInter = sSes.trial_intervals.inter[0:1].copy()
    splitTime = np.ceil(trainingInter[0,1]*0.8)
    trainingInter[0,1] = splitTime
    print("training interval:",trainingInter)
    testInter = sSes.trial_intervals.inter[0:1].copy()
    testInter[0,0] = splitTime
    testInter
    print("testInter:",testInter)
    
    ## get X and y for our model, for both training and test
    train_grid_coord, train_time = transform_xy_to_grid_representation(sessionName,cells,sSessions,trainingInter)    
    train_ifr = get_ifr_for_model(sessionName,cells,sSessions,trainingInter).T
    if train_grid_coord.shape[0] != train_ifr.shape[0]:
        raise ValueError("train_grid_coord should have the same shape[0] as train_ifr")
    test_grid_coord, test_time = transform_xy_to_grid_representation(sessionName,cells,sSessions,testInter)
    test_ifr = get_ifr_for_model(sessionName,cells,sSessions,testInter).T    
    if test_grid_coord.shape[0] != test_ifr.shape[0]:
        raise ValueError("test_grid_coord should have the same shape[0] as test_ifr")

    print("Data points in training set:",train_grid_coord.shape[0])
    print("Data points in test set:",test_grid_coord.shape[0])
        
    
    # plot the path to confirm that the data are as expected
    rowSize,colSize= figurePanelDefaultSize()
    ncols=2
    nrows=1
    fig = plt.figure(figsize=(ncols*colSize, nrows*rowSize), constrained_layout=True) # create a figure
    mainSpec = fig.add_gridspec(ncols=ncols, nrows=nrows)

    ax = fig.add_subplot(mainSpec[0])
    ap.set_intervals(trainingInter)
    ax.plot(ap.pose[:,1],ap.pose[:,2])
    ax.set_title("Training set",fontsize=9)

    ax = fig.add_subplot(mainSpec[1])
    ap.set_intervals(testInter)
    ax.plot(ap.pose[:,1],ap.pose[:,2])
    ax.set_title("Test set",fontsize=9)
    plt.show()
    
    print(sessionName, ":", len(cg.gc_list), "grid cells")
    # HYPERPARAMETERS
    config = {"seq_length":20,
              "n_cells":train_ifr.shape[1],
              "hidden_size" :256,
              "num_layers" : 2,
              "num_outputs" : 4, # sin and cos of v0 and v1
              "learning_rate" : 0.001,
              "batch_size" :64,
              "num_epochs": 1}
    
    # train the model
    df, model = train_rnn_with_config(config,
                                  train_grid_coord,train_time,train_ifr,
                                  test_grid_coord,test_time, test_ifr,sSes)
    
    
    # save the model
    fn = sSes.fileBase+".grid_cell_rnn_model.pt"
    print("saving the model state_dict to",fn)
    torch.save(model.state_dict(), fn)
        




class NeuralDataset(torch.utils.data.Dataset):
    """
    Represent our pose and neural data.
    
    """
    def __init__(self, ifr, pose, time, seq_length,ifr_normalization_means=None,ifr_normalization_stds=None):
        """
        ifr: instantaneous firing rate
        pose: position of the animal
        seq_length: length of the data passed to the network
        """
        super(NeuralDataset, self).__init__()
        self.ifr = ifr.astype(np.float32)
        self.pose = pose.astype(np.float32)
        self.time = time.astype(np.float32)
        self.seq_length = seq_length
        
        self.ifr_normalization_means=ifr_normalization_means
        self.ifr_normalization_stds=ifr_normalization_stds
        
        self.normalize_ifr()
        
        self.validIndices = np.argwhere(~np.isnan(self.pose[:,0]))
        self.validIndices = self.validIndices[self.validIndices>seq_length] # make sure we have enough neural dat leading to the pose
   
        
    def normalize_ifr(self):
        """
        Set the mean of each neuron to 0 and std to 1
        Neural networks work best with inputs in this range
        Set maximal values at -5.0 and 5 to avoid extreme data points
        
        ###########
        # warning #
        ###########
        
        In some situation, you should use the normalization of the training set to normalize your test set.
        For instance, if the test set is very short, you might have a very poor estimate of the mean and std, or the std might be undefined if a neuron is silent.
        """
        if self.ifr_normalization_means is None:
            self.ifr_normalization_means = self.ifr.mean(axis=0)
            self.ifr_normalization_stds = self.ifr.std(axis=0)
            
        self.ifr = (self.ifr-np.expand_dims(self.ifr_normalization_means,0))/np.expand_dims(self.ifr_normalization_stds,axis=0)
        self.ifr[self.ifr> 5.0] = 5.0
        self.ifr[self.ifr< -5.0] = -5.0
        
        
    def __len__(self):
        return len(self.validIndices)
    
    def __getitem__(self,index):
        """
        Function to get an item from the dataset
        
        Returns pose, neural data
        
        """
        neuralData = self.ifr[self.validIndices[index]-self.seq_length:self.validIndices[index],:]
        pose = self.pose[self.validIndices[index]:self.validIndices[index]+1,:] #
        time = self.time[self.validIndices[index]:self.validIndices[index]+1]
        
        return torch.from_numpy(neuralData), torch.from_numpy(pose).squeeze(), torch.from_numpy(time) # we only need one channel for the mask


    
class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_outputs, sequence_length,device):
        super(LSTM,self).__init__()
        """
        For more information about nn.LSTM -> https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        """
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size,hidden_size, num_layers, batch_first=True)
        # input : batch_size x sequence x features
        self.device = device
        self.fc = torch.nn.Linear(hidden_size*sequence_length, num_outputs) # if you onely want to use the last hidden state (hidden_state,num_classes)
        
    def forward(self,x):
        
        h0 =  torch.zeros(self.num_layers,x.size(0), self.hidden_size).to(self.device)
        c0 =  torch.zeros(self.num_layers,x.size(0), self.hidden_size).to(self.device) 
        out, _ = self.lstm(x,(h0,c0))
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out) #if you want to use only the last hidden state, remove previous line, # out = self.fc(out[:,-1,:])
        
        return out
    
def lossOnTestDataset(model,test_data_loader,device,loss_fn):
    model.eval()
    loss_test = 0
    with torch.no_grad():
        for imgs, labels, time in test_data_loader: # mini-batches with data loader, imgs is sequences of brain activity, labels is position of mouse
            imgs = imgs.to(device=device) # batch x chan x 28 x 28 to batch x 28 x 28
            labels = labels.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs,labels)
            loss_test += loss.item()
    a = model.train()
    return loss_test/len(test_data_loader)

def training_loop(n_epochs,
                 optimizer,
                 model,
                 loss_fn,
                 train_data_loader,
                 test_data_loader,
                 config,
                  device,
                 verbose=False):
    
    if verbose:
        print("Training starting at {}".format(datetime.datetime.now()))
    testLoss =  lossOnTestDataset(model,test_data_loader,device,loss_fn)
    trainLoss = lossOnTestDataset(model,train_data_loader,device,loss_fn)
    if verbose:
        print("Test loss without training: {}".format(testLoss))
    
    df = pd.DataFrame({"epochs": [0],
                       "seq_length": config["seq_length"],
                       "n_cells": config["n_cells"],
                       "hidden_size": config["hidden_size"],
                       "num_layers": config["num_layers"],
                      "learning_rate": config["learning_rate"],
                      "batch_size": config["batch_size"],
                      "train_loss": trainLoss,
                      "test_loss": testLoss})

    for epoch in range(1,n_epochs+1):
        loss_train = 0
        for imgs, labels, time in train_data_loader: # mini-batches with data loader, imgs is sequences of brain activity, labels is position of mouse
            imgs = imgs.to(device=device) # batch x chan x 28 x 28 to batch x 28 x 28
            labels = labels.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            
        testLoss = lossOnTestDataset(model,test_data_loader,device,loss_fn)
        if verbose:
            print("{} Epoch: {}/{}, Training loss: {}, Testing loss: {}".format(datetime.datetime.now(),epoch,n_epochs,loss_train/len(train_data_loader), testLoss))
        df1 = pd.DataFrame({"epochs": [epoch],
                       "seq_length": config["seq_length"],
                       "n_cells": config["n_cells"],
                       "hidden_size": config["hidden_size"],
                       "num_layers": config["num_layers"],
                      "learning_rate": config["learning_rate"],
                      "batch_size": config["batch_size"],
                      "train_loss": loss_train/len(train_data_loader),
                           "test_loss": testLoss})
        
        df = pd.concat([df, df1])
    return df



    
def train_rnn_with_config(config,train_grid_coord, train_time, train_ifr, test_grid_coord, test_time,test_ifr, sSes, verbose=True):
    """
    This only takes the first 4 columns in train_grid_coord and test_grid_coord (v0, v1)
    """
    
    print(datetime.datetime.now(),config)
    
    train_dataset = NeuralDataset(ifr =train_ifr[:,:config["n_cells"]], pose=train_grid_coord[:,0:4], time=train_time,  seq_length=config["seq_length"])
    ifr_normalization_means = train_dataset.ifr_normalization_means
    ifr_normalization_stds = train_dataset.ifr_normalization_stds
    
    myDict = {"ifr_normalization_means": ifr_normalization_means,
              "ifr_normalization_stds": ifr_normalization_stds}
    
    fn = sSes.fileBase+".rnn_ifr_normalization.pkl"
    print("Saving:",fn)
    with open(fn, 'wb') as handle:
        pickle.dump(myDict, handle)
    
    
    test_dataset = NeuralDataset(ifr =test_ifr[:,:config["n_cells"]], 
                             pose=test_grid_coord[:,0:4], 
                                 time= test_time,
                             seq_length=config["seq_length"],
                             ifr_normalization_means=ifr_normalization_means,
                             ifr_normalization_stds=ifr_normalization_stds)
    
    
    
    
    train_data_loader = torch.utils.data.DataLoader(train_dataset,batch_size=config["batch_size"] , num_workers=2, shuffle=True, pin_memory=True) # to load batches
    test_data_loader = torch.utils.data.DataLoader(test_dataset,batch_size=config["batch_size"] , num_workers=2, shuffle=False, pin_memory=False) # to load batches

    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cup'))
    #print("Device available:", device)
    model = LSTM(config["n_cells"], config["hidden_size"], config["num_layers"], config["num_outputs"],config["seq_length"],device=device).to(device)
    #print(model)
     
    #neurals, poses = next(iter(train_data_loader))
    
    #print("neurals.shape:",neurals.shape)
    
    #model(neurals.to(device)) # the unsqueeze is to simulate a batch

    
    optimizer = torch.optim.Adam(model.parameters(),lr=config["learning_rate"])
    loss_fn = torch.nn.MSELoss() # 

    model.train()

    df = training_loop(n_epochs=config["num_epochs"],
                 optimizer=optimizer,
                 model = model,
                 loss_fn = loss_fn,
                 train_data_loader=train_data_loader,
                 test_data_loader=test_data_loader,
                 config=config,
                 device = device,
                 verbose=True)
    return df, model

def mvt_direction_error(mvt, predMvt):
    """
    Calculate the movement direction error between real and predicted movement direction
    
    mvt is v1
    predMvt is v2
    
    Gives the angle between the vectors in a counterclockwise direction from v1 to v2. 
    
    v1 is [x1,y1] and v2 is [x2,y2]
    
    atan2d(x1*y2-y1*x2,x1*x2+y1*y2)
    
    If that angle would exceed 180 degrees, then the angle is measured in the clockwise direction but given a negative value. 
    In other words, the output of 'atan2d' always ranges from -180 to +180 degrees.
    
    
    If the mvt is (1,0) and predMvt is (1,1), we get a positive error (0.785)
    If the mvt is (1,0) and predMvt is (1,-1), we get a positive error (-0.785)
    
    If predMvt is (0,-1) and mvt is (1,-1), we get a negative error (-0.785)
    If predMvt is (0,-1) and mvt is (-1,-1), we get a positive error (0.785)
    
    
    """
    # getting the direction of our vectors
    mvtDir = np.arctan2(mvt[:,1],mvt[:,0])
    predMvtDir = np.arctan2(predMvt[:,1],predMvt[:,0])
    
    # use cos to get the x and sin to get the y
    mvtDirError = np.arctan2(np.cos(mvtDir)*np.sin(predMvtDir)-np.sin(mvtDir)*np.cos(predMvtDir),
                             np.cos(mvtDir)*np.cos(predMvtDir)+np.sin(mvtDir)*np.sin(predMvtDir))
    return mvtDirError


def mean_mvt_direction_error(mvtDirError):
    """
    Calculate the mean direction of the mvt direction error
    """
    xMean = np.nanmean(np.cos(mvtDirError))
    yMean = np.nanmean(np.sin(mvtDirError)) 
    return np.arctan2(yMean, xMean)

def vl_mvt_direction_error(mvtDirError):
    """
    Calculate the mean direction of the mvt direction error
    """
    xMean = np.mean(np.cos(mvtDirError))
    yMean = np.mean(np.sin(mvtDirError)) 
    return np.sqrt(xMean*xMean+yMean*yMean)



def get_grid_param_transformation(sSes):
    """
    Get the dictionary used to go from v0,v1,v2 to x,y
    """
    fn = sSes.fileBase+".grid_cell_parameters.pkl"
    #print("Loading:",fn)
    with open(fn, 'rb') as fp: 
        params = pickle.load(fp)
    oriFlexible = np.stack([p["grid_param_model_flexible"]["orientation"] for p in params])
    periodFlexible = np.stack([p["grid_param_model_flexible"]["period"] for p in params])
    grid_param_transformation = {
        "period": np.median(periodFlexible,axis=0),
        "orientation": np.median(oriFlexible,axis=0),
    }
    return grid_param_transformation

def get_grid_rotation_df(myProject):
    """
    Return the dataframe that has the information concerning the rotation of the grid module between the OF and light AutoPI and between OF and dark AutoPI
    """
    fn = myProject.dataPath+"/results/gridRotationDf.csv"
    return pd.read_csv(fn)






def visualize_transform_xy_to_grid_representation(sessionName, cells, sSessions):
    """
    Function to visualize the 2D to 3D representation of grid cells.
        
    It calls `transform_xy_to_grid_representation()` to do the transformation
    
    """
   
    cm_per_bin = 3
    xy_range=np.array([[-50,-90],[50,60]])
    
    sSes, ap, apSim, cg = load_session_files_for_modelling(sessionName,cells,sSessions)
    
    # Get intervals of random foraging to work with
    firstTrialInter = sSes.trial_intervals.inter[0:1]
    print("Intervals:",firstTrialInter)
    
    # get gridSpacePose
    gridSpacePose, time = transform_xy_to_grid_representation(sessionName,cells,sSessions,firstTrialInter)
    
    # get ifrs
    ifrs = get_ifr_for_model(sessionName,cells,sSessions,firstTrialInter)
    ap.set_intervals(firstTrialInter)
    
    print("gridSpacePose.shape:",gridSpacePose.shape)
    print("ifrs.shape:",ifrs.shape)
  
    for n in cg.gc_list:
        n.spike_train.set_intervals(firstTrialInter)
        n.spatial_properties.firing_rate_map_2d(cm_per_bin =cm_per_bin, smoothing_sigma_cm = 5, smoothing=True,xy_range=xy_range)
    
    n = cg.gc_list[0]
    
    # remove invalid values from the pose
    poseLong = ap.pose[:,1:3]
    keepIndices = ~np.any(np.isnan(poseLong),1)
    pose = poseLong[keepIndices]
    
    
    # plot the c0,c1,c2 angle as a funtion of xy position
    rowSize,colSize= 1.6,1.6
    ncols=3
    nrows=2
    fig = plt.figure(figsize=(ncols*colSize, nrows*rowSize), constrained_layout=True) # create a figure
    mainSpec = fig.add_gridspec(ncols=ncols, nrows=nrows)
    for i,label in enumerate(["cos v0","sin v0", "cos v1", "sin v1", "cos v2", "sin v2"]):
        c= int(i / 2)
        r= int(i % 2)
        ax = fig.add_subplot(mainSpec[r,c])
        ax.scatter(pose[:,0],pose[:,1],c = gridSpacePose[:,i], s=1)
        ax.set_xlabel("x position (cm)")
        ax.set_xlabel("y position (cm)")
        ax.set_title(label,fontsize=9)
        ax.set_xlim(-50,50)
        ax.set_ylim(-50,50)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_aspect('equal')
    plt.show()
    
    
    
    v0 = np.arctan2(gridSpacePose[:,1],gridSpacePose[:,0])
    v1 = np.arctan2(gridSpacePose[:,3],gridSpacePose[:,2])
    v2 = np.arctan2(gridSpacePose[:,5],gridSpacePose[:,4])

    ## plot grid cells rate in xy and c0-c1-c2 space
    rowSize,colSize= 1.6,1.6 #figurePanelDefaultSize()
    ncols=2
    nrows=len(cg.gc_list)
    fig = plt.figure(figsize=(ncols*colSize, nrows*rowSize), constrained_layout=True) # create a figure
    mainSpec = fig.add_gridspec(ncols=ncols, nrows=nrows)
    smoothing = 2
    
    for i,n in enumerate(cg.gc_list):
        ax = fig.add_subplot(mainSpec[i,0])
        plotMap(ax, n.spatial_properties.firing_rate_map[1:-1,10:],title = "{:.2f} Hz".format(np.nanmax(n.spatial_properties.firing_rate_map[1:-1,10:])),
               titleFontSize=9)

        ax = fig.add_subplot(mainSpec[i,1])
        myMap,edges_x,edges_y,_ = stats.binned_statistic_2d(np.squeeze(v0),np.squeeze(v1),ifrs[i], statistic='mean',bins=20)
        myMap = ndimage.gaussian_filter(myMap,sigma=smoothing,mode="wrap")
        ax.imshow(myMap.T,origin="lower",extent=[-np.pi,np.pi,-np.pi,np.pi])
        ax.set_title("{:.2f} Hz".format(np.nanmax(myMap)),fontsize=9)
        ax.set_xlabel("v0")
        ax.set_ylabel("v1")
        ax.set_xticks(ticks=[-np.pi, 0, np.pi])
        ax.set_xticklabels([r'-$\pi$', "0", "$\pi$"])
        ax.set_yticks(ticks=[-np.pi, 0, np.pi])
        ax.set_yticklabels([r'-$\pi$', "0", "$\pi$"])
        
    plt.show()
    
def transform_xy_to_grid_representation(sessionName,cells,sSessions,intervals,verbose=False):
    """
    Function used to transform the xy to v0,v1,v2 coordinate system for a session
    The direction and period of v0,v1,v2 depend on the grid cells recorded in this session.
    We get the grid parameter from .grid_cells_parameters.pkl
    """
    # load session files
    sSes, ap, apSim, cg = load_session_files_for_modelling(sessionName,cells,sSessions)
    
    # load grid cell parameters
    fn = sSes.fileBase+".grid_cell_parameters.pkl"
    if verbose:
        print("Loading:",fn)
    with open(fn, 'rb') as fp: 
        params = pickle.load(fp)
    oriRigid = np.stack([p["grid_param_model_rigid"]["orientation"] for p in params])
    oriFlexible = np.stack([p["grid_param_model_flexible"]["orientation"] for p in params])
    periodRigid = np.stack([p["grid_param_model_rigid"]["period"] for p in params])
    periodFlexible = np.stack([p["grid_param_model_flexible"]["period"] for p in params])


    grid_param_transformation = {
    "period": np.median(periodFlexible,axis=0),
    "orientation": np.median(oriFlexible,axis=0),
    }

    if verbose:
        print(grid_param_transformation)
    
    # set the intervals
    ap.set_intervals(intervals)
    
    #print("First time of the pose:",ap.pose[0,0])
    # remove invalid values from the pose
    poseLong = ap.pose[:,0:3]
    keepIndices = ~np.any(np.isnan(poseLong),1)
    pose = poseLong[keepIndices,1:3]
    time = poseLong[keepIndices,0]
    
    gt = gridTransformation(period = grid_param_transformation["period"],
                       orientation = grid_param_transformation["orientation"])
    
    gridSpacePose = gt.poseToGridSpace(pose=pose)
    
    return gridSpacePose,time



def open_field_reconstruction(sessionName,cells,sSessions):
    sSes, ap, apSim, cg = load_session_files_for_modelling(sessionName,cells,sSessions)
    print(sSes.desen)
    print(sSes.trial_intervals.inter)
    nTrials = len(sSes.desen)
    
    testInter = sSes.trial_intervals.inter[(nTrials-1):(nTrials)].copy()
    #testInter[0,1] = testInter[0,0]+(testInter[0,1]-testInter[0,0])*0.8
    #testInter[0,1] = testInter[0,0]+(testInter[0,1]-testInter[0,0])*0.8
    #testInter = sSes.trial_intervals.inter[0:1].copy()
    print("testInter:",testInter)
    
    predMvt, mvt , time = predict_movement_path_one_interval(sessionName,cells,sSessions,testInter)
    
    fn = sSes.fileBase+".open_field_reconstruction.pkl"
    print("Saving:",fn," with mvt, predMvt and time")
    with open(fn, 'wb') as handle:
        pickle.dump({"mvt":mvt,"predMvt":predMvt,"time":time}, handle)
    
    
    rowSize,colSize= 3,3
    ncols=3
    nrows=1
    fig = plt.figure(figsize=(ncols*colSize, nrows*rowSize), constrained_layout=True) # create a figure
    mainSpec = fig.add_gridspec(ncols=ncols, nrows=nrows)
    start=1
    end=-1 

    ax = fig.add_subplot(mainSpec[0])
    ax.plot(np.cumsum(mvt[start:end,0]),np.cumsum(mvt[start:end,1]))
    ax.set_title("Original path",fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")

    ax = fig.add_subplot(mainSpec[1])
    ax.plot(np.cumsum(predMvt[start:end,0]),np.cumsum(predMvt[start:end,1]))
    ax.set_title("Reconstructed path",fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")


    mvtDirError = mvt_direction_error(mvt[start:end], predMvt[start:end])
    meanMvtDirError = mean_mvt_direction_error(mvtDirError)
    vl = vl_mvt_direction_error(mvtDirError)

    ax = fig.add_subplot(mainSpec[2])
    hist = ax.hist(mvtDirError,bins=np.linspace(-np.pi,np.pi,10))
    ax.plot([meanMvtDirError,meanMvtDirError],[0,np.max(hist[0])])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Mvt direction error (radian)")
    ax.set_ylabel("Frequency")
    ax.set_title("Mvt direction error\n mean:{:.2f} vl:{:.2f}".format(meanMvtDirError,vl),fontsize=9)


    fn = "/home/kevin/Downloads/rec_path.png"
    print("Saving",fn)
    plt.savefig(fn)
    plt.close()
    
    secPerSamples=1/50
    rowSize,colSize= 3,3
    ncols=3
    nrows=1
    fig = plt.figure(figsize=(ncols*colSize, nrows*rowSize), constrained_layout=True) # create a figure
    mainSpec = fig.add_gridspec(ncols=ncols, nrows=nrows)

    speed = np.sqrt(np.sum(mvt**2,axis=1))/secPerSamples
    predSpeed = np.sqrt(np.sum(predMvt**2,axis=1))/secPerSamples


    secPerSamples=1/50

    ax = fig.add_subplot(mainSpec[0])
    ax.hist(speed,bins = np.linspace(0,80,20))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Running speed (cm/sec)")
    ax.set_ylabel("Frequency")

    ax = fig.add_subplot(mainSpec[1])
    ax.hist(predSpeed,bins = np.linspace(0,80,20))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Predicted running speed (cm/sec)")
    ax.set_ylabel("Frequency")

    ax = fig.add_subplot(mainSpec[2])
    ax.hist(predSpeed-speed,bins=np.linspace(-60,60,20))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Speed error (pred-real)")
    ax.set_ylabel("Frequency")
    
    rowSize,colSize= 3,3
    ncols=3
    nrows=1
    fig = plt.figure(figsize=(ncols*colSize, nrows*rowSize), constrained_layout=True) # create a figure
    mainSpec = fig.add_gridspec(ncols=ncols, nrows=nrows)


    mvtDirError = mvt_direction_error(mvt, predMvt)
    meanMvtDirError = mean_mvt_direction_error(mvtDirError)
    vl = vl_mvt_direction_error(mvtDirError)
    

    ax = fig.add_subplot(mainSpec[0])
    hist=ax.hist(mvtDirError,bins=np.linspace(-np.pi,np.pi,20))
    ax.plot([meanMvtDirError,meanMvtDirError],[0,np.max(hist[0])])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Mvt direction error (radian)")
    ax.set_ylabel("Frequency")
    ax.set_title("Mvt direction error\n mean:{:.2f} vl:{:.2f}".format(meanMvtDirError,vl),fontsize=9)

    
    speedThreshold = 20

    
    meanMvtDirError = mean_mvt_direction_error(mvtDirError[speed<speedThreshold])
    vl = vl_mvt_direction_error(mvtDirError[speed<speedThreshold])   
    ax = fig.add_subplot(mainSpec[1])
    hist=ax.hist(mvtDirError[speed<speedThreshold],bins=np.linspace(-np.pi,np.pi,20))
    ax.plot([meanMvtDirError,meanMvtDirError],[0,np.max(hist[0])])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Mvt direction error (radian)")
    ax.set_ylabel("Frequency")
    ax.set_title("Mvt direction error \n speed < {}\n mean:{:.2f} vl:{:.2f}".format(speedThreshold,meanMvtDirError,vl),fontsize=9)
    
    meanMvtDirError = mean_mvt_direction_error(mvtDirError[speed>speedThreshold])
    vl = vl_mvt_direction_error(mvtDirError[speed>speedThreshold])   
    ax = fig.add_subplot(mainSpec[2])
    hist=ax.hist(mvtDirError[speed>speedThreshold],bins=np.linspace(-np.pi,np.pi,20))
    ax.plot([meanMvtDirError,meanMvtDirError],[0,np.max(hist[0])])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Mvt direction error (radian)")
    ax.set_ylabel("Frequency")
    ax.set_title("Mvt direction error \n speed > {}\n mean:{:.2f} vl:{:.2f}".format(speedThreshold,meanMvtDirError,vl),fontsize=9)



    fn = "/home/kevin/Downloads/rec_direction_error.png"
    print("Saving",fn)
    plt.savefig(fn)
    plt.close()

    

def predict_movement_path_one_interval(sessionName,cells,sSessions,interval,plot=False):
    sSes, ap, apSim, cg = load_session_files_for_modelling(sessionName,cells,sSessions)
    
    test_grid_coord,test_time = transform_xy_to_grid_representation(sessionName,cells,sSessions,interval)
    test_ifr = get_ifr_for_model(sessionName,cells,sSessions,interval).T   
    
    if test_grid_coord.shape[0] != test_ifr.shape[0]:
        raise ValueError("test_grid_coord should have the same shape[0] as test_ifr")

    print("Data points in test set:",test_grid_coord.shape[0])
    
    # plot the path to confirm that the data are as expected
    if plot == True:
        rowSize,colSize= figurePanelDefaultSize()
        ncols=1
        nrows=1
        fig = plt.figure(figsize=(ncols*colSize, nrows*rowSize), constrained_layout=True) # create a figure
        mainSpec = fig.add_gridspec(ncols=ncols, nrows=nrows)

        ax = fig.add_subplot(mainSpec[0])
        ap.set_intervals(interval)
        ax.plot(ap.pose[:,1],ap.pose[:,2])
        ax.set_title("Test set",fontsize=9)
        plt.show()

    # HYPERPARAMETERS
    config = {"seq_length":20,
          "n_cells":test_ifr.shape[1],
          "hidden_size" :256,
          "num_layers" : 2,
          "num_outputs" : 4,
          "learning_rate" : 0.001,
          "batch_size" :64,
          "num_epochs": 1}
    
    ## load normalization parameters for test dataset
    fn = sSes.fileBase+".rnn_ifr_normalization.pkl"
    print("Loading:",fn)
    normali = pickle.load(open(fn,"rb"))

    
    # create test_dataset
    test_dataset = NeuralDataset(ifr =test_ifr[:,:config["n_cells"]], 
                             pose=test_grid_coord[:,:config["num_outputs"]], 
                             time = test_time,
                             seq_length=config["seq_length"],
                             ifr_normalization_means=normali["ifr_normalization_means"],
                             ifr_normalization_stds=normali["ifr_normalization_stds"])
    # data loader
    test_data_loader = torch.utils.data.DataLoader(test_dataset,batch_size=config["batch_size"] , 
                                                   num_workers=2, shuffle=False, pin_memory=False) # to load batches
    
    # create our model
    device = 'cpu'
    model = LSTM(config["n_cells"], config["hidden_size"], config["num_layers"], config["num_outputs"],config["seq_length"],device=device).to(device)
    
    # load the model state
    fn = sSes.fileBase+".grid_cell_rnn_model.pt"
    
    model.load_state_dict(torch.load(fn))
    model.eval()
    
    
    predictions = np.empty((len(test_dataset),config["num_outputs"]))
    loadedLabels = np.empty((len(test_dataset),config["num_outputs"]))
    recTime = np.empty((len(test_dataset),1))
    
    i = 0

    with torch.no_grad():
        for imgs, labels, time in test_data_loader: # mini-batches with data loader, imgs is sequences of brain activity, labels is position of mouse
            imgs = imgs.to(device=device) # batch x chan x 28 x 28 to batch x 28 x 28
            outputs = model(imgs)
            outputs = outputs.to(device="cpu")
            loadedLabels[i:(i+labels.shape[0]),:] = labels
            predictions[i:(i+labels.shape[0]),:] = outputs
            recTime[i:(i+labels.shape[0])] = time
            i=i+labels.shape[0]
    
    # get the parameters to transform grid cell space into x,y space
    fn = sSes.fileBase+".grid_cell_parameters.pkl"
    print("Loading:",fn)
    with open(fn, 'rb') as fp: 
        params = pickle.load(fp)
    oriFlexible = np.stack([p["grid_param_model_flexible"]["orientation"] for p in params])
    periodFlexible = np.stack([p["grid_param_model_flexible"]["period"] for p in params])
    grid_param_transformation = {
        "period": np.median(periodFlexible,axis=0),
        "orientation": np.median(oriFlexible,axis=0),
    }
    
    # save the prediction in the v3 coordinate system
    fn = sSes.fileBase+".open_field_grid_space_reconstruction.pkl"
    print("Saving:",fn, "with predictions, labels and time")
    with open(fn, 'wb') as handle:
        pickle.dump({"predictions":predictions,"labels":loadedLabels,"time":recTime}, handle)
    
    
    gt = gridTransformation(period = grid_param_transformation["period"],
                       orientation = grid_param_transformation["orientation"])
    
    predMvt = gt.gridSpaceToMovementPath(predictions)
    mvt =     gt.gridSpaceToMovementPath(loadedLabels)

    if plot == True:
        rowSize,colSize= 3,3
        ncols=2
        nrows=1
        fig = plt.figure(figsize=(ncols*colSize, nrows*rowSize), constrained_layout=True) # create a figure
        mainSpec = fig.add_gridspec(ncols=2, nrows=1)

        ax = fig.add_subplot(mainSpec[0])
        ax.plot(np.cumsum(mvt[1:,0]),np.cumsum(mvt[1:,1]))
        ax.set_title("Original path",fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")

        ax = fig.add_subplot(mainSpec[1])
        ax.plot(np.cumsum(predMvt[1:,0]),np.cumsum(predMvt[1:,1]))
        ax.set_title("Reconstructed path ({} gc)".format(config["n_cells"]),fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        plt.show()
    
    return predMvt, mvt, recTime
 
    
  


class gridTransformation:
    """
    Class to do deal with the change in coordinate systems when working with grid cell.
    The two main coordinate systems are 1) x,y position of the animal and 2) a 3D representation of position along the 3 axis of the grid pattern v0,v1,v2
    
    Comes from the lstm_spike_to_position repository
    
    Arguments:
    period: length in cm of the underlying oscillation creating the grid pattern. 
    orientation: orientation of the 3 axes of the grid cells in radians.
    """
    
    def __init__(self, period=np.array([40,40,40]),
                       orientation=np.array([0,np.pi/3,np.pi/3*2])):
        self.set_grid_parameters(period,orientation)
        
        
    def set_grid_parameters(self,period,orientation):
        """
        Function to call if you want to modify the parameters of the grid pattern.
        """
        if period.shape[0] != 3:
            raise ValueError("period should have a length of 3")
        if orientation.shape[0] != 3:
            raise ValueError("orientation should have a length of 3")

        self.period = period
        self.orientation = orientation
        self.set_rotation_matrices()
        
        
    def set_rotation_matrices(self):
        # these matrices are used to project the position of the animal in x,y coordinate system on a grid axis
        # Imagine that a position (x,y) for the mouse is a vector. We rotate the vector by the orientation of the grid axis. We can get the x component of that rotated vector.
        self.Rx0 = np.array([[np.cos(-self.orientation[0])],[-np.sin(-self.orientation[0])]]) # minus sign because we want to rotate the inverse of the angle to bring it back to 1,0 
        self.Rx1 = np.array([[np.cos(-self.orientation[1])],[-np.sin(-self.orientation[1])]])
        self.Rx2 = np.array([[np.cos(-self.orientation[2])],[-np.sin(-self.orientation[2])]])
        
       
    def __str__(self):
        return f'Grid period: {self.period}, Grid orientation: {self.orientation}\nRx0: {self.Rx0}\nRx1: {self.Rx1}\nRx2: {self.Rx2}' 
        
    def __rep__(self):
        return f'Grid period: {self.period}, Grid orientation: {self.orientation}'
   
    def poseToGridSpace(self, pose, angularCoordinate = True, returnCosSin = True):
        """
        Function to transfrom the x,y position of the mouse to 
        a position within the internal representation of grid cells.
        
        The internal representation can be set to be circular (default) or linear
        
        Argument:
        pose: x,y position of the mouse. np.array with 2 columns
        angularCoordinate: whether to transform the position along the grid axes to circular data
        returnCosSin: if angularCoordinate is True, whether to return the cos and sin of the 3 grid angles or return the angle itself.
        """
        
        if pose.shape[1] != 2:
            raise ValueError("pose should have 2 columns")
        
        d0 = pose @ self.Rx0 # d for distance along an axis
        d1 = pose @ self.Rx1
        d2 = pose @ self.Rx2

        if angularCoordinate:
            
            # this turn the disance along each axes as an angle that range from a distance of 0 to the period of the grid pattern.
            c0 = self.length_to_angle(d0, self.period[0]) # c because it is now circular
            c1 = self.length_to_angle(d1, self.period[1]) 
            c2 = self.length_to_angle(d2, self.period[2])
        

            if returnCosSin:
                # we get the cos and sin of each angle as it is easier to work with when training models.
                c0c = np.cos(c0)
                c0s = np.sin(c0)
                c1c = np.cos(c1)
                c1s = np.sin(c1)
                c2c = np.cos(c2)
                c2s = np.sin(c2)
                # return the cos and sin of our 3 angles
                return np.stack([c0c.flatten(),
                                 c0s.flatten(),
                                 c1c.flatten(),
                                 c1s.flatten(),
                                 c2c.flatten(),
                                 c2s.flatten()]).T
            else: # return the angle in radians
                res = np.stack([c0,c1,c2]).T
                return np.squeeze(res,0)
        else:
            # we want linear data normalize by the period
            
            return np.squeeze(np.stack([d0/self.period[0],
                                        d1/self.period[1],
                                        d2/self.period[2]])).T
        
        
    def length_to_angle(self,x,period):
        """
        Function to turn a distance along a grid axis to an angle. 
        A distance of 1 period with be equal to 2*pi. 
        If the animal goes further along the axis, it is like going back to 0, i.e., this is a circular axis.
        
        We use np.arctan2(np.sin, np.cos) instead of a modulo because a modulo function is not differentialble.
        """
        xr = x/period*np.pi*2
        return (np.arctan2(np.sin(xr), np.cos(xr)))
    

    def gridSpaceToMovementPath(self, grid_coord):
        """
        Function to go from grid cell coordinate (2 angles) to movement path

        gridSpace is a representation of the internal activity of the grid manifold. It has 3 dimensions that are circular. But we are only using 2 dimensions here
        When the active representation in grid space changes, we can transform this into movement in the real world.
        We don't know the absolute position of the animal, but we can recreate the movement path.

        We use 2 of the 3 components of the grid space to reconstruct the movement path.
        For each time sample, we know the movement in the grid cells space along these 2 directions.
        If we know that the mouse moved 2 cm along the first grid vector, the mouse can be at any position on a line that passes by 2*unitvector0 and is perpendicular to unitvector0
        If we know that the mouse moved 3 cm along the second grid vector, the mouse can be at any position on a line that passes by 3*unitvector1 and is perpendicular to unitvector1
        We just find the intersection of the two lines to know the movement of the mouse in x,y space.


        Arguments:
        grid_coord: is a 2D numpy array with the cos and sin component of the first 2 axes of the grid (4 columns)
        """

        # get angle from the cos and sin components
        ga0 = np.arctan2(grid_coord[:,1],grid_coord[:,0])
        ga1 = np.arctan2(grid_coord[:,3],grid_coord[:,2])

        # get how many cm per radian

        cm_per_radian = self.period/(2*np.pi)

        # get the movement along the 3 vector of the grid
        dga0=self.mvtFromAngle(ga0,cm_per_radian[0])
        dga1=self.mvtFromAngle(ga1,cm_per_radian[1])


        # unit vector and unit vector perpendicular to the grid module orientation vectors
        uv0 = np.array([[np.cos(self.orientation[0]),np.sin(self.orientation[0])]]) # unit vector v0
        puv0 = np.array([[np.cos(self.orientation[0]+np.pi/2),np.sin(self.orientation[0]+np.pi/2)]]) # unit vector perpendicular to uv0
        uv1 = np.array([[np.cos(self.orientation[1]),np.sin(self.orientation[1])]]) # unit vector v1
        puv1 = np.array([[np.cos(self.orientation[1]+np.pi/2),np.sin(self.orientation[1]+np.pi/2)]]) # unit vector perpendicular to uv1

        # two points in the x,y coordinate system that are on a line perpendicular to v0
        p1 = np.expand_dims(dga0,1)*uv0 # x,y coordinate of movement along v0
        p2 = p1+ puv0 # a second x,y coordinate that is p1 plus a vector perpendicular to uv0

        # two points in the x,y coordinate system that are on a line perpendicular to v1
        p3 = np.expand_dims(dga1,1)*uv1 # coordinate of the point 1 on line 1
        p4 = p3+ puv1 # coordinate of point 2 on line 1

        # find the intersection between 2 lines, using 2 points that are part of line 1 and 2 points that are part of line 2
        # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
        px_num = (p1[:,0]*p2[:,1] - p1[:,1]*p2[:,0]) * (p3[:,0]-p4[:,0]) - (p1[:,0]-p2[:,0]) * (p3[:,0]*p4[:,1] - p3[:,1]*p4[:,0]) 
        px_den = ((p1[:,0]-p2[:,0]) * (p3[:,1]-p4[:,1]) - (p1[:,1]-p2[:,1]) * (p3[:,0]-p4[:,0]))
        reconstructedX = px_num/px_den
        py_num = (p1[:,0]*p2[:,1] - p1[:,1]*p2[:,0]) * (p3[:,1]-p4[:,1]) - (p1[:,1]-p2[:,1]) * (p3[:,0]*p4[:,1] - p3[:,1]*p4[:,0]) 
        py_den = ((p1[:,0]-p2[:,0]) * (p3[:,1]-p4[:,1]) - (p1[:,1]-p2[:,1]) * (p3[:,0]-p4[:,0]))
        reconstructedY = py_num/py_den

        return np.stack([reconstructedX,reconstructedY]).T

    def mvtFromAngle(self,ga,cm_per_radian):
        """
        Go from an angle in the one grid coordinate (one of the 3 axes) to a change in position along this axis
        """
        dga = np.diff(ga,prepend=np.nan) # this is the change in the angle
        dga = np.where(dga>np.pi,dga-2*np.pi,dga) # correct for positive jumps because of circular data
        dga = np.where(dga<-np.pi,dga+2*np.pi,dga) # correct for negative jumps
        dga = dga* cm_per_radian # transform from change in angle to change in cm
        return dga
    
    

def fit_von_mises_1d(x):
    """
    Function to fit a von mises to 1d circular data.
    We get the circular mean as starting point.
    
    Argument
    x: angles in radians
    
    returns
    kappa,mu,scale
    """
    cm = np.arctan2(np.mean(np.sin(x)),np.mean(np.cos(x)))
    
    return vonmises.fit(x, fscale=1, loc=cm) # we have a fixed scale of 1 and initial mu of cm

def BVM_nonNormalized(x,mu,k,A):
    """
    Vectorized probability density function of a bivariate von Mises distribution
    
    Based on Kurz and Hanebeck, Toroidal information fusion based on the bivariate von Mises distribution
    https://isas.iar.kit.edu/pdf/MFI15_Kurz.pdf
    
    Arguments:
    x: input data points as a numpy array with 2 columns.
    mu: means of the distribution as a numpy array with 2 elements
    k: kappas of the distribution as a numpy array with 2 elements
    A: parameter influencing the correlation of the two variables, np.array of shape 2,2
    
    """
    
    E = np.array([np.cos(x[:,0]-mu[0]),
              np.sin(x[:,0]-mu[0])])

    E = np.expand_dims(E,0)
    ET = np.transpose(E,(2,0,1))
    
    F = np.array([np.cos(x[:,1]-mu[1]),
              np.sin(x[:,1]-mu[1])])
    F = np.expand_dims(F,0)
    F = np.transpose(F,(2,1,0))
    
    ETAF = np.squeeze(ET @ A @ F)
    
    return np.exp(k[0] * np.cos(x[:,0]-mu[0])  +  k[1] * np.cos(x[:,1]-mu[1]) + ETAF)

def BVM_normalized(x,mu,k,A,step=0.1):
    """
    Vectorized probability density function of a bivariate von Mises distribution
    
    Based on Kurz and Hanebeck, Toroidal information fusion based on the bivariate von Mises distribution
    https://isas.iar.kit.edu/pdf/MFI15_Kurz.pdf
    
    Arguments:
    x: input data points as a numpy array with 2 columns.
    mu: means of the distribution as a numpy array with 2 elements
    k: kappas of the distribution as a numpy array with 2 elements
    A: parameter influencing the correlation of the two variables, np.array of shape 2,2
    step: step for the numerical integration
    """
    
    # normalization, area under the plane (or torus)
    xx = np.arange(-np.pi,np.pi,step)
    xs,ys = np.meshgrid(xx,xx)
    X = np.stack([xs.flatten(),ys.flatten()]).T
    prob = BVM_nonNormalized(X,mu,k,A)
    C = np.sum(prob)*step**2
    
    # probability for our input data points
    prob = BVM_nonNormalized(x,mu,k,A)
    return (1/C) * prob


def objective_function(x0,X):
    
    mu = np.array([x0[0],x0[1]])
    k =  np.array([x0[2],x0[3]])
    A = np.array([[x0[4],x0[5]],[x0[6],x0[7]]])
    
    
    # probability of each observation
    prob = BVM_normalized(X,mu,k,A)
    
    # log likelihood is sum of log(prob)
    ll = np.sum(np.log(prob))
    return -ll


def fit_bivariate_von_mises(X):
    """
    Fit a bivariate von Mises model to bivariate circular data
    
    Argument
    X: np.array with 2 columns, the angles are in radians.
    
    Return kappas, mu and A as a tupple
    """
    
    # get inital values to start fitting
    k0,m0,_ = fit_von_mises_1d(X[:,0])
    k1,m1,_ = fit_von_mises_1d(X[:,1])


    x0 = np.array([m0, m1, 1, 1 ,0,0,0,0]) # parameters to optimize
    bounds = ((-np.pi,np.pi),(-np.pi,np.pi),(0.0001,20),(0.0001,20), (0,0.5),(0,0.5),(0,0.5),(0,0.5))
    results = minimize(objective_function,x0,args=(X),bounds=bounds)

    resM = np.array([results.x[0],results.x[1]])
    resK =  np.array([results.x[2],results.x[3]])
    resA = results.x[4:8].reshape((2,2))

    return resK, resM, resA