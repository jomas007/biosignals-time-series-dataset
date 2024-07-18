import os, pandas as pd

def create_vd_info(VIDEO_ID, DURATION_ORIGINAL, FPS_ORIGINAL, SHAPE_ORIGINAL, FOLDER_PREFIX, DESTINATION_FOLDER_PATH, VD_INFO_FILE_NAME, VIDEO_NAME, ORIGIN_VID):
    # Variables Calculation
    TOTAL_FRAMES = int(DURATION_ORIGINAL * FPS_ORIGINAL)
    TIME_STEP_FRAME = (1/FPS_ORIGINAL)
    PROCESS_STATUS = 'I'
    HEIGHT_VID = SHAPE_ORIGINAL[0]
    WIDTH_VID = SHAPE_ORIGINAL[1]

    # Video info DF
    current_video_in = pd.DataFrame([( VIDEO_ID, ORIGIN_VID, PROCESS_STATUS, VIDEO_NAME, 
                                   HEIGHT_VID, WIDTH_VID, DURATION_ORIGINAL, FPS_ORIGINAL, TOTAL_FRAMES, TIME_STEP_FRAME)],
              columns=('video_id', 'origin_vid', 'process_status', 'link_video', 
                       'height_vid', 'width_vid', 'duration_vid', 'fps_vid', 'total_frames', 'time_step_fr' ))
    
    # Create the Video ID Folder, Increment 000000 left side
    VIDEO_ID_STR_FOLDER =  DESTINATION_FOLDER_PATH + os.sep + FOLDER_PREFIX + str(VIDEO_ID).zfill(10)  + os.sep

    if not os.path.exists(VIDEO_ID_STR_FOLDER):      
        os.mkdir( VIDEO_ID_STR_FOLDER )
    
    # Save the video information into the destination folder
    current_video_in.to_csv(VIDEO_ID_STR_FOLDER + VD_INFO_FILE_NAME)

