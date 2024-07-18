import numpy as np, pandas as pd, os

def measure_two_points_from_csv (path_csv_file, LANDMK_INIT_PT, LANDMK_ENDD_PT, measure_name):
    
    # Read the Dataframe from CSV
    csv_data_frame_in = pd.read_csv(path_csv_file)
    
    if "Unnamed: 0" in csv_data_frame_in.columns:
    # Remove the Unamed columns
        csv_data_frame_in.drop(columns=["Unnamed: 0"], inplace=True)
    
    # Calculate the Number Of Frames
    NUMBER_OF_FRAMES_IN = len(csv_data_frame_in)
    
    # Measurements path
    CSV_IN_MEASUREMENTS = []
    
    # Iterate and Measure
    for idx in range(0, NUMBER_OF_FRAMES_IN):
        
        # INIT Point XY
        POINT_INIT_XYLMK = np.asarray(eval(csv_data_frame_in[str(LANDMK_INIT_PT)][idx]))
        
        # END Point XY
        POINT_ENDD_XYLMK = np.asarray(eval(csv_data_frame_in[str(LANDMK_ENDD_PT)][idx]))
        
        # Measure
        distance_open_mouth_basic = np.linalg.norm(POINT_ENDD_XYLMK - POINT_INIT_XYLMK) 
        
        # Append in the array
        CSV_IN_MEASUREMENTS.append(distance_open_mouth_basic)
    
    # Create a DataFrame
    MEASURE_RESULTS_DATA_FRAME = pd.DataFrame(CSV_IN_MEASUREMENTS, columns=[measure_name])
    return MEASURE_RESULTS_DATA_FRAME

def measure_vertical_two_arrays_mean (path_csv_file, POINT_ARRAY_INIT, POINT_ARRAY_ENDD, measure_name):
    
    # Read the Dataframe from CSV
    csv_data_frame_in = pd.read_csv(path_csv_file)
    
    # Remove the Unamed columns
    if "Unnamed: 0" in csv_data_frame_in.columns:
        csv_data_frame_in.drop(columns=["Unnamed: 0"], inplace=True)
    
    # Calculate the Number Of Frames
    NUMBER_OF_FRAMES_IN = len(csv_data_frame_in)
    
    # Measurements path
    CSV_IN_MEASUREMENTS = []
    
    # Iterate and Measure
    for idx in range(0, NUMBER_OF_FRAMES_IN):

        # INIT Point Y
        VALUE_Y_INIT = []
        for curr_colect in POINT_ARRAY_INIT:
            basic_to_add = np.asarray(eval(csv_data_frame_in[str(curr_colect)][idx]))
            VALUE_Y_INIT.append (basic_to_add[1])

        # END Point Y
        VALUE_Y_ENDD = []
        for curr_colect in POINT_ARRAY_ENDD:
            basic_to_add = np.asarray(eval(csv_data_frame_in[str(curr_colect)][idx]))
            VALUE_Y_ENDD.append (basic_to_add[1])

        # Calculate the mean
        mean_initial = np.mean(VALUE_Y_INIT)
        mean_endd = np.mean(VALUE_Y_ENDD)

        # Measure
        distance_open_mouth_basic = abs(mean_initial - mean_endd)
        
        # Append in the array
        CSV_IN_MEASUREMENTS.append(distance_open_mouth_basic)
    
    # Create a DataFrame
    MEASURE_RESULTS_DATA_FRAME = pd.DataFrame(CSV_IN_MEASUREMENTS, columns=[measure_name])
    return MEASURE_RESULTS_DATA_FRAME