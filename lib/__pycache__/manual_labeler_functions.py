import matplotlib.pyplot as plt, ast, plotly.express as px, webbrowser, json, pandas as pd, os, cv2

def READ_CSV_FILE(base_path, file_name):
    file_path = os.path.join(base_path, file_name)
    
    # Read the Dataframe from CSV
    current_dt = pd.read_csv(file_path)

    # Remove the Unamed columns
    current_dt.drop(columns=["Unnamed: 0"], inplace=True)
    return current_dt

def CREATE_LABELED_INDEX(FILE_LOCATION_TREE_DLOCAL, VD_INFO_FILE_NAME = 'VD_INFO.CSV', VD_LABEL_FILE_NAME = 'VD_LABELED_L0.CSV'):
    INDEX_REF_DT = pd.DataFrame()
    
    for measure_path in FILE_LOCATION_TREE_DLOCAL:
        current_path = os.path.dirname(measure_path)
        path_vd_info = os.path.join(current_path, VD_INFO_FILE_NAME)

        # Read the Dataframe from CSV
        current_vd_info = pd.read_csv(path_vd_info)
        
        # Remove the Unamed columns
        current_vd_info.drop(columns=["Unnamed: 0"], inplace=True)
        SELECT_DT = current_vd_info[['video_id', 'link_video', 'duration_vid', 'total_frames']].copy()
        path_vd_label = os.path.join(current_path, VD_LABEL_FILE_NAME)
        if os.path.exists(path_vd_label):
            SELECT_DT.loc[:,'label_file_exist']=1
        else:
            SELECT_DT.loc[:,'label_file_exist']=0
        SELECT_DT['path'] = current_path
        INDEX_REF_DT = pd.concat([INDEX_REF_DT, SELECT_DT], ignore_index=True)
    INDEX_REF_DT = INDEX_REF_DT.set_index(pd.Index(INDEX_REF_DT['video_id']))
    INDEX_REF_DT.drop(columns=["video_id"], inplace=True)
        
    return INDEX_REF_DT

def separate_intervals(lst):
    # Sort the list to ensure numbers are in ascending order
    sorted_list = sorted(lst)
    
    # Initialize the intervals list
    intervals = []
    
    # Initialize the start and end of interval
    start_interval = sorted_list[0]
    end_interval = sorted_list[0]
    
    # Iterate over the sorted list
    for num in sorted_list[1:]:
        
        # If the current number is equal to the previous number + 1, continue the interval
        if num == end_interval + 1:
            end_interval = num
        
        # If not, the interval has ended, so add it to the intervals list and start a new interval
        else:
            intervals.append((start_interval, end_interval))
            start_interval = num
            end_interval = num
    
    # Add the last interval to the intervals list
    intervals.append((start_interval, end_interval))
    
    return intervals

# Function to plot a graph with markers for the classes
def PLOT_CLASS_GRAPH(VD_LABELED_DT, VD_MEASURE_DT_V2, class_in, start_frame=None, end_frame=None):
    fonte = {'family': "Arial", 'color': 'black', 'weight': 'bold', 'size': 10}
    
    get_measur = GET_MEASURES_FROM_CLASS (VD_LABELED_DT, class_in)
    frames_f_class = GET_FRAMES_FROM_CLASS(VD_LABELED_DT, class_in)
    frames_f_class= separate_intervals(frames_f_class)
    PLOT_DT = VD_MEASURE_DT_V2[get_measur].copy()

    if start_frame is not None and end_frame is not None:
        PLOT_DT = PLOT_DT[start_frame:end_frame+1]
        
    # Plot graph
    fig, ax = plt.subplots(figsize=(9, 3))
    
    ax.plot(PLOT_DT.index, PLOT_DT, label=get_measur)
    
    for interval in frames_f_class:
        ax.fill_between(interval, 0, 1, alpha=0.2, transform=ax.get_xaxis_transform(), label=f'{class_in}: {interval}')
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel('Amplitude (pixel)', fontdict=fonte)
    plt.xlabel('Frame number', fontdict=fonte)
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig('graph.png')
    plt.show()

def LOAD_VIDEO_FRAMES(path, start_frame=None, end_frame=None, EXTRACT_ALL_FRAMES=True):
    MAX_FRAMES = 1000 # Max number of frames to load
    frames={}
        
    cap = cv2.VideoCapture(path)
    
    if EXTRACT_ALL_FRAMES:
        frame_number = 0
        
        if not cap.isOpened():
            print("Error opening video")
        else:
            while True:
                ret, frame = cap.read()  # Cap frame
                if not ret:
                    break
        
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames[frame_number] = frame_rgb
                frame_number += 1

                if frame_number > MAX_FRAMES: # Break if the maximum number of frames is reached
                    break
                
            cap.release()

    else:
        frame_number = start_frame
        
        if not cap.isOpened():
            print("Error opening video")
        else:
            while True:
                ret, frame = cap.read()  # Cap frame
                if not ret:
                    print("Error capturing frame")
                    break
        
                if start_frame <= frame_number <= end_frame:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames[frame_number] = frame_rgb
                
                frame_number += 1
        
                if frame_number > end_frame:
                    break
                
                if frame_number - start_frame > MAX_FRAMES: # Break if the maximum number of frames is reached
                    break
            cap.release()
            
    return frames

def DISPLAY_FRAMES(frames, start_frame=None, end_frame=None, max_col=5, DISPLAY_ALL_FRAMES=False):
    if DISPLAY_ALL_FRAMES:
        frames_range = frames
    else:
        frames_range = {numero_frame: frame for numero_frame, frame in frames.items() if start_frame <= numero_frame <= end_frame}
    
    # Loop to display images 
    i=1
    n_rows = (len(frames_range) + 4) // max_col
    fig_width = 15
    fig_height  = 0.5 * (end_frame-start_frame+1)
    plt.figure(figsize=(fig_width, fig_height))
    for frame_number, frame in frames_range.items():
        plt.subplot(n_rows, max_col, i)
        plt.imshow(frame)
        plt.text(0,-10,f"frame: {frame_number}")
        plt.axis('off')
        i+=1

def GET_ALL_CLASSES (data_frame_out):
    
    # Function Read Labels
    general_dict = {}
   
    # Check Unique Labels 
    for current_df in data_frame_out['label_measures']:
        general_dict.update((ast.literal_eval(current_df)))
        
    return list(general_dict.keys())

def GET_FRAMES_FROM_CLASS (data_frame_in, class_in):
    classes_to_frames_dict = {}
    for index, row in data_frame_in.iterrows():
        curr_dict = ast.literal_eval(row['label_measures'])
        curr_labels = list(curr_dict.keys())
        for label in curr_labels:
            if label not in classes_to_frames_dict:
                classes_to_frames_dict[label] = []
            classes_to_frames_dict[label].append(row['frame_seq'])
    return classes_to_frames_dict[class_in]

def GET_MEASURES_FROM_CLASS (data_frame_out, class_in):
    general_dict = {}
    
    # Check Unique Labels 
    for current_df in data_frame_out['label_measures']:
        general_dict.update((ast.literal_eval(current_df)))
        
    return general_dict[class_in]

def plot_time_series(time, values, label):
    plt.figure(figsize=(10,6))
    plt.plot(time, values)
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Value", fontsize=20)
    plt.title(label, fontsize=20)
    plt.grid(True)

def UPDATE_LABEL_DF (init_lab, endd_lab, label_name_in, label_measur_in, data_frame_in):
    

    # Check if END is Greater than Length
    for index_x in range(init_lab, endd_lab+1):
        idx_retur_str = data_frame_in['label_measures'][index_x]
        dicct_current = ast.literal_eval(idx_retur_str)
    
        # Insert Updating DICT
        dicct_current.update ({label_name_in: label_measur_in})
        
        # Put Dict into the Current DATA FRAME
        data_frame_in.loc[index_x, 'label_measures'] = str(dicct_current)
       
    return data_frame_in