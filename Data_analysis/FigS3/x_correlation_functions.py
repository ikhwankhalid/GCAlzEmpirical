import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import ndimage

def plotMap(ax,myMap,title="",titleY=0.95,titleFontSize=10,transpose=True,cmap="jet"):
    """
    Plot one 2D map
    """
    if transpose:
        ax.imshow(myMap.T,origin="lower",cmap=cmap,interpolation=None, vmin=0)
    else:
        ax.imshow(myMap,origin="lower",cmap=cmap,interpolation=None, vmin=0)
    ax.set_title(title,y=titleY,fontsize=titleFontSize)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

def spatialCrosscorrelationRemovePadding(cc,extraPadding=0):
    """
    Remove columns and rows that only have np.nan in the spatial crosscorrelation map, while keeping the center at the center of the map.
    This only remove rows before the first rows with a valid value. Same for columns.
    
    Usefull when you want to plot only data with valid data.
    
    Argument
    extraPadding: how many rows and columns of padding to keep.
    """
    x1=(np.nansum(cc,axis=1)!=0).argmax(axis=0)
    x2=np.flip(np.nansum(cc,axis=1)!=0).argmax(axis=0)
    xcut=np.min([x1-extraPadding,x2-extraPadding])
    y1=(np.nansum(cc,axis=0)!=0).argmax(axis=0)
    y2=np.flip(np.nansum(cc,axis=0)!=0).argmax(axis=0)
    ycut=np.min([y1-extraPadding,y2-extraPadding])    
    ccSmall = cc[xcut:-xcut,ycut:-ycut]
    return ccSmall

  
def crosscorrelation_map_stats(cc,cm_per_bin, field_detection_neighborhood_size=3.5, field_detection_max_min_threshold=0.1, field_detection_min_peak_value=0.2):
    """
    Function to calculate some stats from a crosscorrelation map
    Based on the location of the peak value or the closest detected field from the center of the map.
    
    Arguments:
    cc: 2D Numpy array containing the spatial crosscorrelation
    cm_per_bin: centimeter per bins in the spatial crosscorrelation
    field_detection_neighborhood_size: argument used by maximum filter
    field_detection_max_min_threshold:
    field_detection_min_peak_value: Minimal value to be included as a peak
    
    Return:
    Location of the peak value in the map.
    Location of the peak value relative to the center of the map
    Location of the peak value relative to the center of the map in cm
    Peak value
    Distance between the peak value and the center of the crosscorrelation
    Direction of the vector going from the center of the crosscorrelation to the peak value.
    
    Location of the field closest to the center
    Location of the field closest to the center relative to the center fo the map
    Location of the field closest to the center relative to the center fo the map in cm
    Distance between the peak value and the center of the crosscorrelation
    
    
    """
    
    ## analysis based on the peak value in the crosscorrelation
    ## crosscorrelation will always have dimension of odd values
    ## for example if 7x7, we want to get 3 x 3
    print("Map shape:", cc.shape)
    
    midPoint=np.array((cc.shape[0]/2-0.5,cc.shape[1]/2-0.5)) 
    print("Mid point:",midPoint)
    
    cxm = cc.copy()
    cxm[np.isnan(cxm)]=np.nanmin(cxm) # set invalid values to the minimal value in the array, won't be detected as peak.
    peakLoc = np.array(np.unravel_index(np.argmax(cxm, axis=None), cxm.shape))
    print("Peak loc:",peakLoc)
    peakValue = cxm[peakLoc[0],peakLoc[1]]
   
    peakLocToCenter = (peakLoc-midPoint)
    peakLocToCenterCm = (peakLoc-midPoint) * cm_per_bin
    offsetCm = np.sqrt(np.sum((peakLocToCenterCm**2)))
    offsetDir = np.arctan2(peakLocToCenterCm[1],peakLocToCenterCm[0])
    
    ## analysis based on field detection, focusing on the field that is the closest to the center of the crosscorrelation
    data = cc.copy()
    data_max = ndimage.filters.maximum_filter(data, field_detection_neighborhood_size) # apply a maximum_filter (set value to the maximal value in a window)
    maxima = (data == data_max) # find the peak pixel in each detected field
    data_min = ndimage.filters.minimum_filter(data, field_detection_neighborhood_size) # get the minimal value in a surrounding window
    diff = ((data_max - data_min) > field_detection_max_min_threshold) # get the difference between field peak and field minimum
    maxima[diff == 0] = 0 # remove field peaks if the difference is not large enough
    maxima[cc<field_detection_min_peak_value] = 0 # remove field peaks if the peak is not larger than min_peak_value

    labels, num_peaks = ndimage.label(maxima) # assigned the peak a value of 1,2,3,4,etc
    slices = ndimage.find_objects(labels) # get the begining and end of each peak
   
    x,xc, y, yc = [],[],[], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        xc.append(round(x_center)-midPoint[1])
        x.append(round(x_center))
        y_center = (dy.start + dy.stop - 1)/2
        yc.append(round(y_center)-midPoint[0])
        y.append(round(y_center))
    distances = []
    for xx,yy in zip(xc,yc):
        d = np.sqrt((xx)**2+(yy)**2)
        distances.append(d)
    fieldIndex = np.argmin(distances) # index of the fields that is the closest to the center 
    print("Detected field peak loc:",  np.array([y[fieldIndex], x[fieldIndex]]))
    
    return peakLoc, peakLocToCenter, peakLocToCenterCm, peakValue, offsetCm, offsetDir, num_peaks, np.array([y[fieldIndex], x[fieldIndex]]),np.array([yc[fieldIndex], xc[fieldIndex]]), np.array([yc[fieldIndex], xc[fieldIndex]])*cm_per_bin , distances[fieldIndex]*cm_per_bin


def plotSpatialCrosscorrelation(ax,cc, removePadding=False,showCenter=True,showPeak=False,useFieldDetection=True,showExpectedPeak=False,xOffset=None,yOffset=None,cm_per_bin=None,title="",
                               field_detection_neighborhood_size=3.5, field_detection_max_min_threshold=0.1, field_detection_min_peak_value=0.2):
    """
    Function to plot spatial crosscorrelation or spike-triggered short-time cross-firing rate map
    
    Arguments:
    ax: axes
    cc: 2D numpy array with the spatial crosscorrelation or ststcfrm
    removePadding: whether to remove padding when plotting. Padding are columns or rows with only invalid data. See spatialCrosscorrelationRemovePadding()
    showCenter: boolean, will plot lines and a dot to show the center of the map
    showPeak: boolean, will plot lines and a dot to show the peak of the map (where the maximal value is)
    useFieldDetection: boolean, if showPeak is True and useFiedlDetection is True, will use field detection to find the closest peak to the center
    showExpectedPeak: boolean, use when debugging, can plot lines and dot to show the expected peak in the map
    xOffset, yOffset: use with showExpectedPeak, this is the xOffset and yOffset 
    
    
    """
    
    cx = cc.copy()
    if removePadding:
        cx=spatialCrosscorrelationRemovePadding(cx,extraPadding=0)
    
    if np.nanmin(cx) < 0: # assumes that the values are from -1 to 1 (Pearson correlation coefficients and not rate), add +1 to values so that negative values are not clamped at 0
        plotMap(ax,cx+1,title="",titleY=0.95,titleFontSize=8) # this plots the transpose of the map
    else :
        plotMap(ax,cx,title="",titleY=0.95,titleFontSize=8) # this plots the transpose of the map
    
    if showCenter:
        # crosscorrelation and autocorrelations always have an odd number of bins
        # for example if 7x7, then mid point should be at index 3,3
        midPoint=(cx.shape[0]/2-0.5,cx.shape[1]/2-0.5)
        #print("Center:",midPoint)
        ax.plot([0,cx.shape[0]], [midPoint[1],midPoint[1]],"--",color="black", alpha=0.5) # horizontal line for the center of the cc
        ax.plot([midPoint[0],midPoint[0]],[0,cx.shape[1]],"--",color="black", alpha=0.5) # vertical line for the center of the cc
        ax.scatter([midPoint[0]],[midPoint[1]],color="black",alpha=0.5,s=1)
    
    if showExpectedPeak: # use for testing the functions
        ax.plot([0,cx.shape[1]], [midPoint[1]+yOffset/cm_per_bin,midPoint[1]+yOffset/cm_per_bin],"--",color="gray", alpha=0.8) # expected offset for testing, horizontal line
        ax.plot([midPoint[0]+xOffset/cm_per_bin, midPoint[0]+xOffset/cm_per_bin],[0,cx.shape[0]],"--",color="gray", alpha=0.8) # expected offset for testing, vertical line
        ax.scatter([midPoint[0]+xOffset/cm_per_bin], [midPoint[1]+yOffset/cm_per_bin],color="red", alpha=0.8) # expected offset for testing, vertical line
    
    if showPeak:
        cxm = cx.copy()
        if useFieldDetection==False:
            
            cxm[np.isnan(cxm)]=np.nanmin(cxm) # set invalid values to the minimal value in the array, won't be detected as peak.
            peakLoc = np.unravel_index(np.argmax(cxm, axis=None), cxm.shape)
            print("Location of the peak value:", peakLoc)
        if useFieldDetection==True:
            _,_,_,_,_,_,_,peakLoc,_,_,_ = crosscorrelation_map_stats(cxm,cm_per_bin=2, field_detection_neighborhood_size=field_detection_neighborhood_size, field_detection_max_min_threshold=field_detection_max_min_threshold, field_detection_min_peak_value=field_detection_min_peak_value)
            print("Location of the detected field peak:", peakLoc)
        
        ax.scatter([peakLoc[0]], [peakLoc[1]],color="pink", alpha=0.8) # real peak in the crosscorrelation
        ax.plot([0,cx.shape[1]], [peakLoc[1],peakLoc[1]],"--",color="pink", alpha=0.8) # expected offset for testing, horizontal line
        ax.plot([peakLoc[0], peakLoc[0]],[0,cx.shape[0]],"--",color="pink", alpha=0.8) # expected offset for testing, vertical line
    ax.set_title(title)  
