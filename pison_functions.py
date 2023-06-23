import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from scipy.fft import fft
from scipy.signal import correlate, find_peaks
from scipy.spatial.distance import euclidean, cdist
from scipy.spatial.transform import Rotation

from fastdtw import fastdtw
from pyts.decomposition import SingularSpectrumAnalysis as ssa

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import train_test_split

from tslearn.metrics import dtw_path
from tslearn.clustering import TimeSeriesKMeans, silhouette_score as ts_silhouette_score

from xgboost import XGBClassifier


import time 


#Visualize data
def plot_wearable(df,wearable_name,my_ax = None,z_score = True, other_traces = True,min_max = False):
    """_Plot a column of data against time, while representing repetition number and body movement

    Args:
        df (pd.DataFrame): contains the wearable dataset as well as time, body movement, and repetition number
        wearable_name (str): Name of column in dataframe to plot
        my_ax (axes, optional): handle to plot the figure on. Defaults to None.
        z_score (bool, optional): Z-score the time series to plot. Defaults to True.
        other_traces (bool, optional): Plot body movement and repetition number on the same axes. Defaults to True.
        min_max (bool, optional): min-max the time series to plot. Defaults to False.
    """

    time = df['Time (ms)']
    wearable = df[wearable_name]

    n_str = ''
    if z_score:
        n_str = 'Z-scored '
        wearable = StandardScaler().fit_transform(wearable.values.reshape(-1,1))
    
    if min_max:
        n_str = 'Min-Maxed '
        wearable = MinMaxScaler().fit_transform(wearable.values.reshape(-1,1))
    
    if my_ax == None: #if no axes passes, create new one
        my_ax = plt.axes()

    my_ax.plot(time,wearable,color = 'black') #z-score data for easier visualization
    
    if other_traces:
        body_movement = df['Body movement label'] 
        rep_num = df['Repetition number']
        my_ax.plot(time,body_movement,color = 'red') #plot the body_movement
        my_ax.scatter(time,np.ones(time.shape)*np.max(wearable),c= rep_num) #plot the rep number

    plt.setp(my_ax,xlabel = 'Time (ms)',ylabel = n_str + wearable_name)

def reset_time(df,time_col = 'Time (ms)'):
    """Reset the time axis so that the first sample is at time 0

    Args:
        df (pandas.DataFrame): DataFrame containing the time axis
        time_col (str, optional): name for the time column. Defaults to 'Time (ms)'.

    Returns:
        pandas.DataFrame: DataFrame with reset time axis
    """
    df[time_col] = df[time_col] - df.loc[0,time_col]
    return df

def identify_async_indx(df):
    """Identify the indices where the signal "jumps" in time based on body movement label. It's easier than with time, since it's at non-zero values.
    If this was different, we could just find peaks of the differential, but let's make it easy for us now.

    Args:
        df (pandas.DataFrame): Dataframe containing the body movement label which we have identified matches with periods of asynchronicity.

    Returns:
        list: contains the indices of time jumps, as well as -1 at the beginning and the last sample to ensure that all our time chunks are well segmented.
    """
    indx_jump = np.argwhere(np.diff(df['Body movement label'])!=0).flatten() #last index of the sample corresponding to a particular body movement
    indx_jump = np.insert(indx_jump,0,-1) #add at the beginning the index -1, since the first segment begins at 0
    indx_jump = np.append(indx_jump,len(df)) #add at the end the last sample

    return indx_jump

def normalize_data(df,col,scaling = 'z-score'):
    """_summary_

    Args:
        df (pandas.DataFrame): DataFrame containing the values to scale
        col (str): column name to scale
        scaling (str, optional): Scaling type. Defaults to 'z-score'.

    Returns:
        pandas.DataFrame: DataFrame containing the scaled values within the same column
    """
    if scaling == 'z-score':
        df[col] = StandardScaler().fit_transform(df[col].values.reshape(-1,1))
    elif scaling == 'min-max':
        df[col] = MinMaxScaler().fit_transform(df[col].values.reshape(-1,1))
    else:
        print('No scaling applied')
    return df

def segment_data(df,win_indx, segment_size = 50):
    """Segregate dataframe into subwindows within continuous data streams.

    Args:
        df (pandas.DataFrame): Dataframe containing all traces of data to segment
        win_indx (list): List of time points corresponding to the last index before a transition point in the time domain
        segment_size (int, optional): Size of the window to segment data. Defaults to 50.

    Returns:
        pandas.DataFrame: DataFrame where each row corresponds to a segmentation, and each column is an array of the values segmented
    """
    seg_df = {}
    for col in df.columns: #iterate through all columns (different features to possibly explore)
        win_vals = []
        for start_indx,end_indx in zip(win_indx[:-1],win_indx[1:]): #iterate through all continuous time series data points
            df_win = df.iloc[start_indx+1:end_indx].copy().reset_index() #first, we are going to window just within the body motion to ensure that we don't capture other body motion datapoints
            df_win['win_id'] = np.arange(len(df_win))//segment_size + 1 #create a new column, corresponding to the window id for this section, in order to efficiently group by
    
            if col in ['Body movement label','Repetition number']:
                #since we saw from the figure above that the transition points match with the transitions of these labels, each window should have only one of these labels
                win_vals.extend(df_win.groupby('win_id')[col].apply(lambda x: np.unique(x)[0]).values)
            else:
                win_vals.extend(df_win.groupby('win_id')[col].apply(np.array).values)
       
        seg_df[col] = win_vals

    seg_df = pd.DataFrame(seg_df) #turn dictionary into dataframe
    seg_df['num_samples'] = seg_df['Time (ms)'].apply(len) #for information purposes, return the number of samples per each window 

    return seg_df

def estimate_sampling_rate(df,async_indx):
    """Estimate sampling rate. Since dataframe has asynchronous periods, we need to ignore those time points.

    Args:
        df (pandas.DataFrame): DataFrame containing time data
        async_indx (list): indices corresponding to the periods where traces become asynchronous

    Returns:
        scalar: Estimated sampling rate in Hz
    """
    tmp = segment_data(df,async_indx,segment_size = 1e10) #a very large window ensures that only one window per continuous period
    fs_indx = []
    for seg_id in range(len(tmp)):
        fs_indx.append(np.mean(np.diff(tmp.loc[seg_id,'Time (ms)'])))
    return 1/np.mean(fs_indx)*1000 #scale from ms to s

def quat_to_euler(df,quaternion_names = ['Quaternion x','Quaternion y','Quaternion z','Quaternion w']):
    """Transform our Quaternion data into Euler rotational  positioning for easier understanding.

    Args:
        df (pandas.DataFrame): Dataframe containing the quaternion data, to add the new Euler rotational data
        quaternion_names (list, optional): Names of the quaternion columns. Defaults to ['Quaternion x','Quaternion y','Quaternion z','Quaternion w'].

    Returns:
        pandas.DataFrame: DataFrame with the removed quaternion columns with the new Euler columns
    """

    
    r = Rotation.from_quat(df[quaternion_names])
    euler_transform = r.as_euler('xyz')
    euler_transform[:,0:2] = euler_transform[:,0:2] * -1 #multiply x and y by -1 since they point West and South, opposite of the gyroscope and accelerometer

    df[['Euler_x','Euler_y','Euler_z']] = euler_transform

    return df

def vis_euler3D(df,euler_names = ['Euler_x','Euler_y','Euler_z'],color = None, change_color = True,ax_handle = None):
    """Visualize Euler Positioning in 3D space.

    Args:
        df (pandas.DataFrame): dataframe containing the euler positionings.
        euler_names (list, optional): column names (x,y,z). Defaults to ['Euler_x','Euler_y','Euler_z'].
        color (tuple, optional): input for scatter for the color to use. Defaults to None.
        change_color (bool, optional): Whether to have the trace change color across the line. Defaults to True.
        ax_handle (plt.axes, optional): axes handle to plot onto. Defaults to None.
    """
    euler_x =df[euler_names[0]] 
    euler_y =df[euler_names[1]] 
    euler_z =df[euler_names[2]] 

    if ax_handle == None:
        plt.figure(figsize = (20,10))
        this_ax = plt.axes(projection = '3d')
    else:
        this_ax = ax_handle

    plt.setp(this_ax,'xlim',[np.min(euler_x),np.max(euler_x)],'ylim',[np.min(euler_y),np.max(euler_y)],'zlim',[np.min(euler_z),np.max(euler_z)],
            'xlabel','East-West','ylabel','North-South','zlabel','Up-Down')

    if color:
        color_pal = color
    elif change_color:
        max_val = euler_x.shape[0]
        color_pal = [(i/max_val, 0, (max_val - i) / max_val) for i in range(max_val)]
    else:
        color_pal = [1,0,0]

    this_ax.scatter(euler_x,euler_y,euler_z,color = color_pal)  

def vis_accel3D(df,accel_names = ['Accelerometer x (m2/s)','Accelerometer y (m2/s)','Accelerometer z (m2/s)'] ,color = None, change_color = True,ax_handle = None):
    """Visualize a 3 dimensional trace of the accelerometers across all 3 axes.

    Args:
        df (pandas.DataFrame): dataframe containing the accelerometer traces.
        accel_names (list, optional): column names (x,y,z). Defaults to ['Accelerometer x (m2/s)','Accelerometer y (m2/s)','Accelerometer z (m2/s)'].
        color (tuple, optional): input for scatter for the color to use. Defaults to None.
        change_color (bool, optional): Whether to have the trace change color across the line. Defaults to True.
        ax_handle (plt.axes, optional): axes handle to plot onto. Defaults to None.
    """
    
    accel_x =df[accel_names[0]] 
    accel_y =df[accel_names[1]] 
    accel_z =df[accel_names[2]] 

    if ax_handle == None:
        plt.figure(figsize = (20,10))
        this_ax = plt.axes(projection = '3d')
    else:
        this_ax = ax_handle

    plt.setp(this_ax,'xlim',[np.min(accel_x),np.max(accel_x)],'ylim',[np.min(accel_y),np.max(accel_y)],'zlim',[np.min(accel_z),np.max(accel_z)],
            'xlabel','East-West','ylabel','North-South','zlabel','Up-Down')

    if color:
        color_pal = color
    elif change_color:
        max_val = accel_x.shape[0]
        color_pal = [(i/max_val, 0, (max_val - i) / max_val) for i in range(max_val)]
    else:
        color_pal = [1,0,0]

    this_ax.plot3D(accel_x,accel_y,accel_z,color = color_pal) 

def vis_accel1D(df,accel_names = ['Accelerometer x (m2/s)','Accelerometer y (m2/s)','Accelerometer z (m2/s)'] ,color = None, change_color = True,ax_handle = None):
    """Visualize an accelerometer trace combining all 3 axes into a single accelerometer magnitude.

    Args:
        df (pandas.DataFrame): dataframe containing the accelerometer traces.
        accel_names (list, optional): column names (x,y,z). Defaults to ['Accelerometer x (m2/s)','Accelerometer y (m2/s)','Accelerometer z (m2/s)'].
        color (tuple, optional): input for scatter for the color to use. Defaults to None.
        change_color (bool, optional): Whether to have the trace change color across the line. Defaults to True.
        ax_handle (plt.axes, optional): axes handle to plot onto. Defaults to None.
    """
    
    accel_x =df[accel_names[0]] 
    accel_y =df[accel_names[1]] 
    accel_z =df[accel_names[2]] 

    tot_acc = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

    if ax_handle == None:
        plt.figure(figsize = (20,10))
        this_ax = plt.axes()
    else:
        this_ax = ax_handle

    plt.setp(this_ax,'ylim',[-1*np.max(tot_acc),np.max(tot_acc)], 'xlabel','Time (s)')

    if color:
        color_pal = color
    elif change_color:
        max_val = tot_acc.shape[0]
        color_pal = [(i/max_val, 0, (max_val - i) / max_val) for i in range(max_val)]
    else:
        color_pal = [1,0,0]

    t_vec = (df['Time (ms)'] - df['Time (ms)'][0])/1000
    this_ax.plot(t_vec,tot_acc,color = color_pal) 

def vis_sensors_filt(df,accel_names = ['Channel 0 HP','Channel 1 HP'] ,color1 = None,color2 = None, change_color = True,ax_handle = None):
    """Visualize the sensor traces for both channels for the high pass traces.

    Args:
        df (pandas.DataFrame): dataframe containing the accelerometer traces.
        accel_names (list, optional): column names sensors data for  high pass data. Defaults to ['Channel 0 HP','Channel 1 HP'].
        color1 (tuple, optional): input for scatter for the color to use for channel 0. Defaults to None.
        color2 (tuple, optional): input for scatter for the color to use for channel 1. Defaults to None.
        change_color (bool, optional): Whether to have the trace change color across the line. Defaults to True.
        ax_handle (plt.axes, optional): axes handle to plot onto. Defaults to None.
    """

    sensor_0_hp =df[accel_names[0]]     
    sensor_1_hp =df[accel_names[1]] 
    

    if ax_handle == None:
        plt.figure(figsize = (20,10))
        this_ax = plt.axes()
    else:
        this_ax = ax_handle

    plt.setp(this_ax, 'xlabel','Time (s)')

    if color1 and color2:
        color_pal1 = color1
        color_pal2 = color2
    elif change_color:
        max_val = df['Time (ms)'].shape[0]
        color_pal1 = [(i/max_val, 0, (max_val - i) / max_val) for i in range(max_val)]
        color_pal2 = [(i/max_val, (max_val - i) / max_val, 0 ) for i in range(max_val)]
    else:
        color_pal1 = [1,0,0]
        color_pal2 = [0,1,0]

    t_vec = (df['Time (ms)'] - df['Time (ms)'][0])/1000

    this_ax.plot(t_vec,sensor_0_hp,color = color_pal1) 
    this_ax.plot(t_vec,sensor_1_hp,color = color_pal2) 
    this_ax.legend(['HP 0','HP 1'])

def vis_sensors_raw(df,accel_names = ['Channel 0 Raw','Channel 1 Raw'] ,color1 = None,color2 = None, change_color = True,ax_handle = None):
    """Visualize the sensor traces for both channels for the raw traces.

    Args:
        df (pandas.DataFrame): dataframe containing the accelerometer traces.
        accel_names (list, optional): column names sensors data for raw data. Defaults to ['Channel 0 Raw','Channel 1 Raw'].
        color1 (tuple, optional): input for scatter for the color to use for channel 0. Defaults to None.
        color2 (tuple, optional): input for scatter for the color to use for channel 1. Defaults to None.
        change_color (bool, optional): Whether to have the trace change color across the line. Defaults to True.
        ax_handle (plt.axes, optional): axes handle to plot onto. Defaults to None.
    """

    sensor_0_raw =df[accel_names[0]] 
    sensor_1_raw =df[accel_names[1]] 

    if ax_handle == None:
        plt.figure(figsize = (20,10))
        this_ax = plt.axes()
    else:
        this_ax = ax_handle

    plt.setp(this_ax, 'xlabel','Time (s)')

    if color1 and color2:
        color_pal1 = color1
        color_pal2 = color2
    elif change_color:
        max_val = df['Time (ms)'].shape[0]
        color_pal1 = [(i/max_val, 0, (max_val - i) / max_val) for i in range(max_val)]
        color_pal2 = [(i/max_val, (max_val - i) / max_val, 0 ) for i in range(max_val)]
    else:
        color_pal1 = [1,0,0]
        color_pal2 = [0,1,0]

    t_vec = (df['Time (ms)'] - df['Time (ms)'][0])/1000

    this_ax.plot(t_vec,sensor_0_raw,color = color_pal1) 
    this_ax.plot(t_vec,sensor_1_raw,color = color_pal2) 
    this_ax.legend(['Raw 0','Raw 1'])

def my_dtw(array_one,array_two,metric=euclidean):
    """Wrapper for fast dtw so that we only return the distance

    Args:
        array_one (np.array): signal one
        array_two (np.array): signal two
        metric (scip.spatial.distance, optional): Metric to use for DTW. Defaults to euclidean.

    Returns:
        float: Dynamic Time Wrapping distance
    """
    distance,_ = fastdtw(array_one,array_two,dist=metric)

    return distance

def estimate_accelerometer_power(df,accel_cols = ['Accelerometer x (m2/s)','Accelerometer y (m2/s)','Accelerometer z (m2/s)']):
    """Compute the magnitude of all the vectors provided (applied to the accelerometer data)

    Args:
        df (pandas.DataFrame): Contains the accelerometer components to aggregate
        accel_cols (list, optional): Column names for the accelerometer components. Defaults to ['Accelerometer x (m2/s)','Accelerometer y (m2/s)','Accelerometer z (m2/s)'].

    Returns:
        pandas.DataFrame: Original dataframe with a new columne "Accelerometer_power"
    """
    df['Accelerometer_power'] = df.apply(lambda row: np.sqrt(sum(row[i]**2 for i in accel_cols)),axis =1)
    return df

def segment_based_on_periods(seg_df,heights = 14,async_indx = [],accel_cols = ['Accelerometer x (m2/s)','Accelerometer y (m2/s)','Accelerometer z (m2/s)']):
    """_summary_

    Args:
        seg_df (pandas.dataFrame): _description_
        heights (int, optional): _description_. Defaults to 14.
        async_indx (list, optional): _description_. Defaults to [].
        accel_cols (list, optional): Column names for the accelerometer components. Defaults to ['Accelerometer x (m2/s)','Accelerometer y (m2/s)','Accelerometer z (m2/s)'].
    Returns:
        _type_: _description_
    """
    seg_df['Accelerometer_power'] = seg_df.apply(lambda row: np.sqrt(sum(row[i]**2 for i in accel_cols)),axis =1)

    segment_pts = []
    for ii in range(seg_df.shape[0]):
        ind,_ = find_peaks(seg_df.loc[ii,'Accelerometer_power'],height = heights)   
        per = np.max(np.abs(np.diff(ind)))
        first_pt = ind[0]
        samp_pts = np.arange(first_pt,len(seg_df.loc[ii,'Accelerometer_power']),per)

        if len(async_indx): 
            samp_pts = async_indx[ii] + samp_pts
        
        segment_pts.extend(samp_pts)

    return segment_pts

def estimate_frequency_spectra(df,col,f_bins = 32):
    """Estimate the frequency spectra for a given column, keeping only the real values.

    Args:
        df (pandas.DataFrame): DataFrame containing the columns to analyze the power spectrum
        col (str): Name of column to analyze
        f_bins (int, optional): length of the fft. Defaults to 32.

    Returns:
        pandas.DataFrame: DataFrame with the new column containing the frequency spectra under "spectra_[column name]"
    """

    #Take the FFT, normalize power, and take the real component only.
    df['spectra_' + col] = df[col].apply(lambda x: np.real(fft(x,n = f_bins)/len(x)))
    return df

def cheap_ssa_analysis(df,col,num_wins = 3):
    """Super cheap way to implement SSA since we iterate as for-loops to assign the vectors.

    Args:
        df (pandas.DataFrame): DataFrame that contains information to decompose
      col (str): Column name to decompose.
      num_wins (int, optional): number of vectors to decompose into. Defaults to 3.

    Returns:
      pandas.DataFrame: DataFrame contained decomposed information
    """
    my_ssa = ssa(window_size = num_wins).fit(df.loc[0,col])
    tmp_out = df[col].apply(lambda x: my_ssa.transform(x.reshape(1,-1)))


    for this_win in range(num_wins):
        tmp = []
        for tmp_out_cell in tmp_out:
            tmp.append(tmp_out_cell[this_win])

        df[col + ' ssa_' + str(this_win)] = tmp

    return df

def univariate_analysis(df,col,stats = {'mean':np.mean, 'std':np.std, 'power' : (lambda x : np.sqrt(np.sum(np.abs(x)**2)))} ):
    """Perform a univariate statistical analysis on a given column

    Args:
        df (pandas.Dataframe): Contains the column to perform univariate analysis
        col (str): Column name
        stats (dict, optional): Statistics to perform on the column, with keys corresponding to the prefix for column names. Defaults to {'mean':np.mean, 'std':np.std, 'power':sqrt(sum(abs(x**2)))}.

    Returns:
        pandas.DataFrame: original dataframe containing new columns with the stats.
    """
    for name,func in stats.items():
        df[name + '_' + col] = df[col].apply(func)
    return df

def bivariate_analysis(df,col1,col2,transforms = {'corr':correlate} , stats = {'mean' :np.mean, 'std':np.std, 'lag' :np.argmax} , joint_stat = {'fast_dtw':my_dtw}):
    """Perform bivariate analyses on sets of columns

    Args:
        df (pandas.Dataframe): Contains the column to perform univariate analysis
        col1 (str): Column name for feature 1
        col2 (str): Column name for feature 2
        transforms (dict, optional): Transformations to perform on joint series. Defaults to {'corr':correlate}.
        stats (dict, optional): Statistics to perform on the transformed series, with keys corresponding to the prefix for column names. Defaults to {'mean' :np.mean, 'std: np.std, 'lag' :np.argmax}.
        joint_stat (dict, optional): Statistics to perform on the joint series. Defaults to {'fast_dtw':my_dtw}.

    Returns:
        pandas.DataFrame: original dataframe containing new columns with the stats.
    """
    transform_dict = {}
    #We are going to transform groups of series into new time series components (i.e., correlation) and then perform certain stats on these transformations.
    #In order to not add new time-series components to our outputs, we are going to store these time series in a dictionary, then perform the stats on them and add
    #these stats to the dictionary. 
    for name,transf in transforms.items(): 
        dict_name = name + '_' + col1 + '_' + col2
        transform_dict[dict_name] =  df.apply(lambda row: transf(row[col1],row[col2]), axis = 1)


        for stat_name, func in stats.items():
            df[stat_name + '_' + dict_name] = transform_dict[dict_name].apply(func)

    #A few statistics depend on both series, so we will do this separately so we don't have to worry about adding more time series etc..
    for name,joint_func in joint_stat.items():
        df[name + '_' + col1 + '_' + col2] = df.apply(lambda row: joint_func(row[col1],row[col2]), axis = 1)
        
    return df

def id_num_clusters(X_df,k_groups = np.arange(2,15),show = True):
    """Identify and visualize the distortion, silhouette score and inertia of clustering the data into various numbers of clusters.

    Args:
        X_df (pandas.DataFrame): DataFrame of features to cluster
        k_groups (np.array, optional): Array of number of clusters to visualize. Defaults to np.arange(1,15).
        show (bool, optional): To display the inertia, distortion and silhouette scores. Defaults to True.
    Returns:
        (list,list,list): distortion, inertia, and silhouette
    """
    inertia,distortion,silhouette = [],[],[]
    
    for k in k_groups:
        kmodel = KMeans(n_clusters = k,n_init = 'auto').fit(X_df)

        distortion.append(np.mean(cdist(X_df, kmodel.cluster_centers_,'euclidean')))
        inertia.append(kmodel.inertia_)
        silhouette.append(silhouette_score(X_df,kmodel.labels_))

    if show:
        plt.figure(figsize = (20,5))
        plt.subplot(131)
        plt.plot(k_groups,distortion,'--o'); plt.xlabel('Number of Clusters'); plt.ylabel('Distortion')
        plt.subplot(132)
        plt.plot(k_groups,inertia,'--o'); plt.xlabel('Number of Clusters'); plt.ylabel('Inertia')
        plt.subplot(133)
        plt.plot(k_groups,silhouette,'-o'); plt.xlabel('Number of Clusters'); plt.ylabel('Silhouette Score')

    return distortion, inertia, silhouette

def Coalesce_Euler3D(df,as_list = True):
    """Function to combine the Euler Trace

    Args:
        df (pandas.DataFrame): DataFrame containing the information for the euler traces spread out.
        as_list (bool, optional): Return values within each cell as a list instead of a np.array. Default True.

    Returns:
        pandas.DataFrame: DataFrame with the combined Euler trace as "Euler Series"
    """
    df['Euler Series'] = df.apply(lambda row: _combine_cols(row['Euler_x'],row['Euler_y'],row['Euler_z'],as_list = as_list),axis = 1)
    return df

def _combine_cols(x_data,y_data,z_data,as_list = True):
    """Combine columns into a single list for using of the dtw algorithms

    Args:
        x_data (np.array): Euler trace for the x direction
        y_data (np.array): Euler trace for the y direction
        z_data (np.array): Euler trace for the z direction
        as_list (bool, optional): Return values within each cell as a list instead of a np.array. Default True.

    Returns:
        list: list of tuples of the euler positioning
    """
    if as_list:
        out_col = [(i,k,j) for i,k,j in zip(x_data,y_data,z_data)]
    else:
        out_col = [np.array([i,k,j]) for i,k,j in zip(x_data,y_data,z_data)]
    return out_col

def dtw_matrix(df,col = 'Euler Series'):
    """Compute the DTW across all samples to generate a distance matrix 

    Args:
        df (pandas.DataFrame): DataFrame to compute the DTW distance matrix

    Returns:
        np.array: matrix containing the distances across all samples.
    """
    dist_mat = []
    for i in range(df.shape[0]):
        tmp = []
        for j in range(df.shape[0]):
            _,dist = dtw_path(df.loc[i,col],df.loc[j,col])
            tmp.append(dist)
        dist_mat.append(tmp)
    return np.array(dist_mat)

def format_column_for_timeseriesK(df,col):
    """Reformat a column to fit the dimensions and requisites of TimeSeriesKMeans.

    Args:
        df (pandas.DataFrame): DataFrame containing the column to reformat
        col (str): Column name to reformat.

    Returns:
        np.array: Array with the reformatted column following the structure required by TimeSeriesKMeans.
    """
    df = df.copy()
    max_samp = np.max(df.num_samples)
    df = df.apply(lambda row: _extend_time(row,col,max_samp),axis =1)
    return np.swapaxes(np.swapaxes(np.dstack(df[col]),1,2),0,1)

def _extend_time(row,col,max_samps,val = np.nan):
    """Subroutine called by format_column_for_timeseriesK to extend the vectors.

    Args:
        row (pandas.Series): Series containing num_samples and the vector that we wish to extend.
        col (str): column to extend
        max_samps (int): Value that we are extending up to.
        val (np.value, optional): Padding value. Defaults to np.nan.

    Returns:
        pandas.Series: Original series but with the 'col' component extended
    """
    extend_by = max_samps - row['num_samples']
    num_dim = len(row[col][0])

    if num_dim == 1:
        vect_extend = np.repeat(val,repeats = extend_by)
    else:
        vect_extend = np.tile(val,reps = [extend_by,num_dim])
    row[col].extend(vect_extend)
    return row

def id_num_time_series_clusters(X_df,k_groups = np.arange(2,10),show = True):
    """Identify and visualize the silhouette and inertia of clustering the data into various numbers of time series clusters.

    Args:
        X_df (pandas.DataFrame): DataFrame of features to cluster
        k_groups (np.array, optional): Array of number of clusters to visualize. Defaults to np.arange(1,15).
        show (bool, optional): To display the inertia, distortion and silhouette scores. Defaults to True.
    
    Returns:
        (list,list): inertia, and silhouette
    
    """
    inertia,silhouette = [],[]
    
    for k in k_groups:
        kmodel = TimeSeriesKMeans(n_clusters = k,init = 'random',metric = 'dtw',max_iter=5).fit(X_df)

        silhouette.append(ts_silhouette_score(X_df,kmodel.labels_))
        inertia.append(kmodel.inertia_)

    if show:
        plt.figure(figsize = (20,5))
        plt.subplot(121)
        plt.plot(k_groups,silhouette,'--o'); plt.xlabel('Number of Clusters'); plt.ylabel('Silhouette score')
        plt.subplot(122)
        plt.plot(k_groups,inertia,'--o'); plt.xlabel('Number of Clusters'); plt.ylabel('Inertia')

    return inertia, silhouette

def my_augmentation(df):
    """Combination of all the data augmentation so that we can repeat it later on more effectively.

    Args:
        df (pandas.DataFrame): DataFrame to turn into a feature matrix for KMeans by augmenting datasets
   
    Returns:
        (pandas.DataFrame, pandas.DataFrame,list) : DataFrame containing only the scalar values and features used for K means, Dataframe containing all of the summary stats and original values., list of columns to drop
    """


    to_drop = ['Time (ms)','Channel 0 Raw','Channel 1 Raw',
        'Quaternion x','Quaternion y','Quaternion z','Quaternion w',
        'num_samples','Repetition number']

    ssa_cols = ['Channel 0 HP','Channel 1 HP','Accelerometer_power','Euler_x','Euler_y','Euler_z']
    freq_cols = ['Channel 0 HP','Channel 1 HP','Accelerometer_power']

    #Create Frequency components
    for col_name in ssa_cols:
        tmp_df = cheap_ssa_analysis(df,col_name)
    for col_name in freq_cols:
        tmp_df = estimate_frequency_spectra(tmp_df,col_name)    
    
    #Summarize time series data with univariate analysis
    stats = {'mean' : np.mean, 'std' : np.std, 'power' :  (lambda x : np.sqrt(np.sum(np.abs(x)**2)))}
    cols1 = ['Accelerometer x (m2/s)','Accelerometer y (m2/s)',
        'Accelerometer z (m2/s)','Gyroscope x (deg/s)','Gyroscope y (deg/s)',
        'Gyroscope z (deg/s)','Euler_x','Euler_y','Euler_z','Channel 1 HP','Channel 0 HP']
    for c in cols1:
        tmp_df = univariate_analysis(df,c,stats)


    #For the spectral analysis, only identify the power
    stats = {'power' :  (lambda x : np.sqrt(np.sum(np.abs(x)**2)))}
    cols2 = ['Euler_x ssa_0', 'Euler_x ssa_1', 'Euler_x ssa_2',
       'Euler_y ssa_0', 'Euler_y ssa_1', 'Euler_y ssa_2',
       'Euler_z ssa_0', 'Euler_z ssa_1', 'Euler_z ssa_2',
       'Channel 0 HP ssa_0', 'Channel 0 HP ssa_1', 'Channel 0 HP ssa_2',
       'spectra_Channel 0 HP', 'Channel 1 HP ssa_0', 'Channel 1 HP ssa_1',
       'Channel 1 HP ssa_2', 'spectra_Channel 1 HP',
       'Accelerometer_power ssa_0', 'Accelerometer_power ssa_1',
       'Accelerometer_power ssa_2', 'spectra_Accelerometer_power']
    for c in cols2:
        tmp_df = univariate_analysis(tmp_df,c,stats)

    cols_to_drop = to_drop + cols1 + cols2 + ['Accelerometer_power','Body movement label']
    X = tmp_df[tmp_df.columns.drop(to_drop).drop(cols1).drop(cols2).drop(['Accelerometer_power','Body movement label'])]
    
    return X,tmp_df,cols_to_drop

def score_model(model,features,labels):
    """Score out the tested model

    Args:
        model (model type): Object containing predict and predict_proba attributes
        features (pandas.DataFrame): Feature set of the testing dataset
        labels (list): Class id for the testing dataset

    Returns:
        float,float: f1 score and roc_score
    """
    preds = model.predict(features)
    f1= f1_score(labels,preds,average = 'macro')
    roc_score = roc_auc_score(labels,model.predict_proba(features),multi_class = 'ovr')

    print(f'f1 score: {f1}, roc_score: {roc_score}')
    return f1,roc_score
