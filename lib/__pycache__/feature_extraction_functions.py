import os, cv2, pandas as pd, datetime, dlib
from imutils import face_utils

import general_functions as gf

def extract_funcion_bz (capture, VIDEO_ID, TIME_STEP_FR):
    LANDMARK_FILE= "." + os.sep + "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(LANDMARK_FILE)

    # Initialize the Data Frame
    VIDEO_STREAMING_PD = pd.DataFrame()

    FRAME_SEQUENCE = 0
    TIME_STEP_CURR = 0

    while capture.isOpened():
        ret, frame = capture.read()
        if ret:  
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            rects = detector(gray_image, 0)
            
            # Loop over the face detections (number of faces detected)
            shape = []
            INSTANCE_REF = 0
            for enumx in rects:
                
                # Determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
                shape = predictor(gray_image, enumx)
                shape = face_utils.shape_to_np(shape)

                # Save with the Main function
                video_parcial = feature_data_frame_l1(VIDEO_ID, FRAME_SEQUENCE, TIME_STEP_CURR, INSTANCE_REF, shape)
                
                # Append the DataFrame
                VIDEO_STREAMING_PD = pd.concat([VIDEO_STREAMING_PD, video_parcial], ignore_index=True)
                
                INSTANCE_REF += 1
            
            # Increment the values streaming
            FRAME_SEQUENCE += 1
            TIME_STEP_CURR += TIME_STEP_FR
        else:
            break
        
    return VIDEO_STREAMING_PD

def feature_data_frame_l1 (VIDEO_ID, FRAME_SEQUENCE, TIME_STEP_CURR, INSTANCE_REF, KEYPOINTS_BZ):
    # Initialize with VIDEO ID
    video_extractor_l1_inst = pd.DataFrame([VIDEO_ID], columns=["video_id"])
     
    video_extractor_l1_inst ['frame_seq'] =  FRAME_SEQUENCE
    video_extractor_l1_inst['time_seconds'] =  TIME_STEP_CURR
    video_extractor_l1_inst['time_datetime'] = str(datetime.timedelta(seconds=TIME_STEP_CURR))
    video_extractor_l1_inst['instance_ref'] = INSTANCE_REF
    
    # Mount the 68 Points in DataFrame
    count_id = 0
    for indx in KEYPOINTS_BZ:
        video_extractor_l1_inst [f"{count_id}"] = str(tuple(indx))
        count_id += 1
    return video_extractor_l1_inst

def FEATURE_EXTRACTOR_L1_ALL (var_FILE_ALL_in, video_origin_path_in):  
    number_of_videos = len(var_FILE_ALL_in)  
    for i, current_compl_path_csv in enumerate(var_FILE_ALL_in):

        print (str(i+1) + " of "+ str(number_of_videos) + " : " + "Starting to process the " + current_compl_path_csv)

        # Read current VID-INFO
        video_info_rest = pd.read_csv(current_compl_path_csv)

        # Collect and organize the Info
        LINK_VIDEO = list(video_info_rest['link_video'])[0]
        TIME_STEP_FR = float(list(video_info_rest['time_step_fr'])[0])
        VIDEO_ID = int(list(video_info_rest['video_id'])[0])
        PROCESS_STATUS = list(video_info_rest['process_status'])[0]
        ORIGIN_VIDD = list(video_info_rest['origin_vid'])[0]
        
        current_folder = os.path.dirname(current_compl_path_csv)

        if PROCESS_STATUS == 'I':
            # Function based on Source YT or DD
            capture = None
            if ORIGIN_VIDD == 'Y':
                VIDEO_RAW_URL = video_origin_path_in + LINK_VIDEO
                capture = cv2.VideoCapture(gf.get_best_url(VIDEO_RAW_URL))
            else:
                VIDEO_RAW_URL = os.path.join(video_origin_path_in, LINK_VIDEO + ".mp4")
                capture = cv2.VideoCapture(VIDEO_RAW_URL)

            if not capture.isOpened():
                print("!!! Erro ao abrir video: " + VIDEO_RAW_URL)

            # Extract the Features
            VIDEO_STREAMING_PD = extract_funcion_bz (capture, VIDEO_ID, TIME_STEP_FR)

            # Final Features File
            VIDEO_STREAMING_PD.to_csv(current_folder + os.sep + "VD_FEATURES_L1" + ".CSV" )
            
            # Remove the Unamed columns
            video_info_rest.drop(columns=["Unnamed: 0"], inplace=True)
            
            # Update the Status to L (Level 1 Landmarks)
            video_info_rest.loc[0, 'process_status'] = 'L'
            
            # Save with the same current CSV path location
            video_info_rest.to_csv(current_compl_path_csv)
        else:
            print("The features from this video has already been extracted")