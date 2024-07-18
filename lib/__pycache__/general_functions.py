import pafy, os, cv2, moviepy.editor, pandas as pd, glob, files_paths as fp, numpy as np

def get_best_url(url):
    try:
        video = pafy.new(url)
        best = video.getbest(preftype="mp4")
        if best is None:
            print(f'ERROR: {url} is unavailable')
            return None
        return best.url
    except Exception as e:
        print(f'ERROR: Unable to fetch URL {url}: {e}')
        return None
    

def get_file_name(file_path, ext):
    basename = os.path.basename(file_path)
    filename = basename.replace(ext, '')
    return filename

def collect_next_video_id(VD_INFO_LOCATION_TREE_LST):
    number_of_videos = [int(file.split(os.sep)[-2].split('_')[-1]) for file in VD_INFO_LOCATION_TREE_LST]
    if len(number_of_videos) == 0:
        next_video = 1
    else: next_video = np.array(number_of_videos).max() + 1
    return next_video

# Collect the Basic informations from the video
def collect_video_info (video_path):
    video_capture = cv2.VideoCapture(video_path)
 
    if video_capture.isOpened():
 
        # Read the video file.
        video_captured_sucessfully, frame = video_capture.read()
 
        # If we got frames, show them.
        if video_captured_sucessfully == True:    
           
            # 1 - Collect the SHAPE
            wh_shape = frame.shape[:2]
            video_capture.release()
           
            # 2 - Collect the FPS and Duration
            vid_ret = moviepy.editor.VideoFileClip(video_path)
            duration_vid = vid_ret.duration
            fps_vid = vid_ret.fps
   
            return video_captured_sucessfully, wh_shape, duration_vid, fps_vid
        else:
            video_capture.release()
            print ("Error -> FUNC-BZ-LIB-> COL-INFO-2")
            return video_captured_sucessfully, None, None, None
    else:  
        video_capture.release()
        print ("Error -> FUNC-BZ-LIB-> COL-INFO-1")
        return False, None, None, None

def READ_CSV_FILE(path, file_name=None):
    if file_name:
        path = os.path.join(path, file_name)
    
    current_dt = pd.read_csv(path)

    if "Unnamed: 0" in current_dt.columns:
        current_dt.drop(columns=["Unnamed: 0"], inplace=True)
    return current_dt

def find_files_in_all_subdirectories(directories, file_name_pattern):
    files = []
    for directory in directories:
        folder_path = os.path.join(directory, '**', file_name_pattern)
        files.extend(glob.iglob(folder_path, recursive=True))
    return sorted(files)

def get_video_path(local, video_id, VD_FILE_NAME):
    if local.upper() == "Y":
        return os.path.join(fp.DATASET_YT, f"VD_Y_{video_id:010}", VD_FILE_NAME)
    elif local.upper() == "D":
        return os.path.join(fp.DATASET_LOCAL, f"VD_D_{video_id:010}", VD_FILE_NAME)
    else:
        return "Invalid Local"

class RangeError(ValueError):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def test_range(start_frame, end_frame):
    if end_frame < start_frame:
        raise RangeError("Error: start frame must be less than end frame.")
    else:
        return True