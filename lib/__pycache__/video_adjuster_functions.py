import numpy as np, os, pandas as pd, datetime, math, cv2

from typing import OrderedDict
from collections import deque

# 03 - Video Adjuster Functions =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def add_fr_exist_column(DATAFRAME, SUMMARY):
    
    # Get index where the discontinuity begin
    frame_desc = np.array((SUMMARY.query('fr_exists == 0').frame_seq_end).reset_index(drop=True),dtype=np.int16)
    frame_desc = frame_desc+1
    index_desc = DATAFRAME[DATAFRAME['frame_seq'].isin(frame_desc)].index
    
    # Set 1 flag in the end frame for all discontinuity  
    DATAFRAME.insert(2, 'gap', np.nan)
    DATAFRAME.loc[index_desc, 'gap'] = 1
    DATAFRAME['gap'] = DATAFRAME['gap'].fillna(0).astype(int)
    
    return DATAFRAME

def find_descontinuites_frames(video_feature_rest, N_FRAMES):    
    
    # Get values of frame_seq
    FRAME_SEQ = np.array(video_feature_rest['frame_seq'], dtype=np.int16)
    
    # Get the maximum value of frame_seq
    frame_seq_max = FRAME_SEQ[-1]
    
    VIDEO_ID = video_feature_rest['video_id'][0]
    
    # Make a dataframe with a video_id column and n rows (n=maximum value of frame_seq)
    FRAMES_DTFRAME = pd.DataFrame(index=pd.RangeIndex(0,frame_seq_max+1,1), dtype=np.int16)
    FRAMES_DTFRAME['video_id']  = VIDEO_ID
    FRAMES_DTFRAME['frame_seq'] = range(0,frame_seq_max+1)
    
    # Fill the collumn fr_exists with 1 in the indexes that the frame_seq number exists
    FRAMES_DTFRAME.loc[FRAME_SEQ, 'fr_exists'] = 1
    
    # Replace NaN values with zeros in 
    FRAMES_DTFRAME['fr_exists'] = FRAMES_DTFRAME['fr_exists'].fillna(0).astype(int)
    
    # Count one each frame that its difference to previus is bigger than 1
    # This value indicates the group this frames belongs
    FRAMES_DTFRAME['total_frames'] = (FRAMES_DTFRAME.fr_exists.diff(1) != 0).cumsum()
    
    # Filter Dataframe to find discontinuous intervals
    SUMMARY = pd.DataFrame({'fr_exists' : FRAMES_DTFRAME.groupby('total_frames').fr_exists.first(),
              'frame_seq_init' : FRAMES_DTFRAME.groupby('total_frames').frame_seq.first(), 
              'frame_seq_end': FRAMES_DTFRAME.groupby('total_frames').frame_seq.last(),
              'total_frames' : FRAMES_DTFRAME.groupby('total_frames').size()}).reset_index(drop=True)
    
    # Select frames to interpolate by threshold
    FRAMES_TO_INTERPOLATE = SUMMARY.query('fr_exists == 0 and frame_seq_init != 0').query(f'total_frames <= {N_FRAMES}').reset_index(drop=True)
    ALL_DISCONTINUITIES = SUMMARY
    
    return FRAMES_DTFRAME, FRAMES_TO_INTERPOLATE, ALL_DISCONTINUITIES

def interpolate_frames(current_path_location, VD_FEATURES_FILE, N_FRAMES=5, OUTPUT_FEATURE_NAME = 'VD_FEATURES_L2.CSV'):
    path_dir = os.path.dirname(current_path_location)

    video_feature_rest = pd.read_csv(os.path.join(path_dir, VD_FEATURES_FILE))
        
    # Remove the Unamed columns
    if 'Unnamed: 0' in video_feature_rest.columns:
        video_feature_rest.drop(columns=['Unnamed: 0'], inplace=True)
    
    # Select lines where dataFrame == 0
    video_feature_rest = video_feature_rest.query('instance_ref == 0')
    
    # Obtain a DataFrame with frame intervals for interpolation
    FRAMES_DTFRAME, FRAMES_TO_INTERPOLATE, ALL_DISCONTINUITIES = find_descontinuites_frames(video_feature_rest, N_FRAMES)
    
    # If there are discontinuities in VD_FEATURES
    if FRAMES_TO_INTERPOLATE.size > 0:
        
        # If a discontinuity is at the video's end.
        if video_feature_rest.iloc[-1].frame_seq == FRAMES_TO_INTERPOLATE.iloc[-1].frame_seq_end:
            FRAMES_TO_INTERPOLATE = FRAMES_TO_INTERPOLATE.drop(-1).reset_index(drop=True)

        # Mount data frame with nan values to interpolate
        video_feature_rest_f = video_feature_rest.set_index('frame_seq')
        
        video_feature_rest_f = video_feature_rest_f.drop(columns = ['video_id'])

        INTERPOLATE_DTFRAME = pd.concat([FRAMES_DTFRAME, video_feature_rest_f], axis=1)
        INTERPOLATE_DTFRAME.at[0, 'time_seconds'] = 0.0
        INTERPOLATE_DTFRAME.time_seconds = INTERPOLATE_DTFRAME.time_seconds.interpolate(method='linear', axis=0)
        INTERPOLATE_DTFRAME.instance_ref = 0
        INTERPOLATE_DTFRAME = INTERPOLATE_DTFRAME.drop(columns = ['fr_exists'])
        INTERPOLATE_DTFRAME = INTERPOLATE_DTFRAME.drop(columns = ['total_frames'])

        # Interpolate Routine
        for info_pointer, fr_to_interpolate_info in FRAMES_TO_INTERPOLATE.iterrows():
            frame_seq_init = fr_to_interpolate_info['frame_seq_init'] - 1
            frame_seq_end = fr_to_interpolate_info['frame_seq_end'] + 2

            subset_landmarks = INTERPOLATE_DTFRAME.iloc[frame_seq_init:frame_seq_end, 5:]

            for landmark_pointer, landmark in subset_landmarks.items():
                points_dt = pd.DataFrame(columns=['x','y'], index=range(len(landmark)), dtype=int)
                points_dt.index = landmark.index

                points_init = eval(landmark[frame_seq_init])
                points_end = eval(landmark[frame_seq_end-1])

                points_dt.iloc[0] = points_init
                points_dt.iloc[-1] = points_end

                result = points_dt.interpolate().drop(axis=1, index=[frame_seq_init, frame_seq_end-1])
                result = result.astype(int, copy=True)

                for i, p in result.iterrows():
                    INTERPOLATE_DTFRAME.at[i, str(landmark_pointer)] = tuple(p)
                    INTERPOLATE_DTFRAME.at[i, 'time_datetime'] = str(datetime.timedelta(seconds=INTERPOLATE_DTFRAME.loc[i]['time_seconds']))
    
        # Fill NaN values in the time_datetime column    
        rows, cols = np.where(pd.isna(INTERPOLATE_DTFRAME[['time_datetime']]))
        for r in rows:
            INTERPOLATE_DTFRAME.at[r, 'time_datetime'] = str(datetime.timedelta(seconds=INTERPOLATE_DTFRAME.loc[r]['time_seconds']))
        
        # Remove rows in ALL_DISCONTINUITIES that were interpolated
        temp_df = pd.merge(ALL_DISCONTINUITIES, FRAMES_TO_INTERPOLATE, indicator=True, how='outer')
        temp_df = temp_df.sort_values(by='frame_seq_init').reset_index(drop=True)
        rows_to_remove = temp_df[temp_df['_merge'] == 'both'].index
        ALL_DISCONTINUITIES = ALL_DISCONTINUITIES.drop(index=rows_to_remove)

        # Drop rows that were not interpolated
        INTERPOLATE_DTFRAME = INTERPOLATE_DTFRAME.dropna().reset_index(drop=True)
    
    # If there are no descontinuities in VD_FEATURES
    else:
        INTERPOLATE_DTFRAME = video_feature_rest.copy()
    
    # Add gap column indicating when occurs a discontinuity
    INTERPOLATE_DTFRAME = add_fr_exist_column(INTERPOLATE_DTFRAME, ALL_DISCONTINUITIES)
    
    # Save CSV
    base_path_folder_output = os.path.join(path_dir, OUTPUT_FEATURE_NAME)
    
    # Generate the OUTPUT CSV in the current folder
    INTERPOLATE_DTFRAME.to_csv(base_path_folder_output)

# 03 - Video Adjuster Spacial Normalization Functions =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def measure_two_points_from_csv_ext (csv_data_frame_in, LANDMK_INIT_PT, LANDMK_ENDD_PT, measure_name):
    CSV_IN_MEASUREMENTS = []
    
    # Iterate and Measure
    for idx in csv_data_frame_in.index:
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

def transform_scale_updown (shape_in, scale_f):
    
    # Define the new shape
    new_shape = []
    
    for vect_v in shape_in:
        vect_v = vect_v * scale_f
        new_shape.append (vect_v)
    return np.array(new_shape)

def tuple_to_np(shape_in):
    shape_out = np.zeros([68,2], dtype=int)
    for i in range(0, len(shape_in)):
        shape_out[i] = np.asarray(eval(shape_in.iloc[i]))
    return shape_out

# Spacial Normalization =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=

FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

def move_to_center_position(shape, point_idx, x_center, y_center):
    selected_point_x, selected_point_y = shape[point_idx]
    new_shape = [(x - (selected_point_x - x_center), y - (selected_point_y - y_center)) for x, y in shape]
    return np.array(new_shape)

def scale_landmarks(scale_factor, landmarks):
    return np.array([(x / scale_factor, y / scale_factor) for x, y in landmarks])

def calculate_distance(landmarks, points):
    x1, y1 = landmarks[points[0]]
    x2, y2 = landmarks[points[1]]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def z_normalization(landmarks, points=[27, 30], divider=30):
    scale_factor = calculate_distance(landmarks, points) / divider
    scaled_landmarks = scale_landmarks(scale_factor, landmarks)
    #print('\nscale_factor', scale_factor)
    return scale_factor, scaled_landmarks

def calculate_eye_center(landmarks, eye_idxs):
    eye_pts = landmarks[eye_idxs[0]:eye_idxs[1]]
    return eye_pts.mean(axis=0).astype("int")

def calculate_angle(point1, point2):
    dY = point2[1] - point1[1]
    dX = point2[0] - point1[0]
    return np.degrees(np.arctan2(dY, dX)) + 180

def normalize_angle(angle):
    return angle - 360 if angle > 180 else angle

def rotate_landmarks(landmarks, angle, center_point):
    angle_rad = math.radians(angle)
    cos_theta, sin_theta = math.cos(angle_rad), math.sin(angle_rad)
    rotated_landmarks = []

    for x, y in landmarks:
        translated_x, translated_y = x - center_point[0], y - center_point[1]
        rotated_x = translated_x * cos_theta - translated_y * sin_theta + center_point[0]
        rotated_y = translated_x * sin_theta + translated_y * cos_theta + center_point[1]
        rotated_landmarks.append([rotated_x, rotated_y])

    return np.array(rotated_landmarks)

def roll_normalization(landmarks):
    left_eye_center = calculate_eye_center(landmarks, FACIAL_LANDMARKS_68_IDXS["left_eye"])
    right_eye_center = calculate_eye_center(landmarks, FACIAL_LANDMARKS_68_IDXS["right_eye"])
    angle = calculate_angle(left_eye_center, right_eye_center)
    desired_rotation = normalize_angle(0 - angle)
    return rotate_landmarks(landmarks, desired_rotation, landmarks[33])

# Statistics generating =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Função para calcular medidas estatísticas
def calcular_medidas_estatisticas(df):
    stats = df.describe().transpose()
    
    stats['variance'] = df.var()
    stats['median'] = df.median()
    stats['range'] = stats['max'] - stats['min']
    stats['cv'] = stats['std'] / stats['mean']  # Coeficiente de variação
    stats['iqr'] = df.quantile(0.75) - df.quantile(0.25)  # Intervalo interquartil
    #stats['mad'] = df.apply(pd.Series.mad)  # Desvio médio absoluto
    stats['skewness'] = df.skew()  # Assimetria
    stats['kurtosis'] = df.kurtosis()  # Curtose
    
    return stats[['mean', 'std', 'median', 'min', 'max', 'variance', 'range', 'cv', 'iqr', 'skewness', 'kurtosis']]

# Função principal para aplicar normalizações e PCA
def aplicar_medidas_estatisticas(df, measure_file):
    file_path = '../outputs/medidas_estatisticas_' + os.path.basename(measure_file) + '.xlsx'
    
    if os.path.exists(file_path):
        # Carregar o conteúdo existente do arquivo Excel
        existing_data = pd.read_excel(file_path, sheet_name=None)
    else:
        existing_data = {}

    # Calcular as medidas estatísticas dos dados originais
    print("Medidas estatísticas dos dados originais:")
    stats_originais = calcular_medidas_estatisticas(df)
    print(stats_originais)

    # Adicionar a primeira coluna com os nomes das colunas do df
    column_names = df.columns
    print("Column names:", column_names)

    type_values = np.resize(column_names, stats_originais.shape[0])  # Ajustar o comprimento de type_values
    print("Type values:", type_values)

    new_rows = stats_originais.copy()
    new_rows.insert(0, 'Type', type_values)

    # Concatenar com os dados existentes, se houver
    if 'Originais' in existing_data:
        original_df = existing_data['Originais']
        # Adicionar uma linha em branco
        blank_row = pd.DataFrame([[''] * original_df.shape[1]], columns=original_df.columns)
        original_df_with_blank = pd.concat([original_df, blank_row], ignore_index=True)
        # Concatenar com as novas estatísticas
        updated_df = pd.concat([original_df_with_blank, new_rows], ignore_index=True)
        existing_data['Originais'] = updated_df
    else:
        existing_data['Originais'] = new_rows

    # Salvar novamente no arquivo Excel
    with pd.ExcelWriter(file_path) as writer:
        for sheet_name, data in existing_data.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print("\nMedidas estatísticas salvas no arquivo Excel.")

# Função principal para aplicar normalizações e PCA
def aplicar_medidas_estatisticas_csv(df, measure_file):
    file_path = os.path.join(os.path.dirname(measure_file), 'STATS_'+os.path.basename(measure_file))
    
    # Calcular as medidas estatísticas dos dados originais
    stats_originais = calcular_medidas_estatisticas(df)
    print(stats_originais)
    stats_originais.to_csv(file_path)
    
    # Adicionar a primeira coluna com os nomes das colunas do df
    #column_names = df.columns
    #print("Column names:", column_names)

    #type_values = np.resize(column_names, stats_originais.shape[0])  # Ajustar o comprimento de type_values
    #print("Type values:", type_values)

    #new_rows = stats_originais.copy()
    #new_rows.insert(0, 'Type', type_values)


    # 

# Smooth landmarks =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class LandmarkSmoother:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.landmark_buffer = deque(maxlen=window_size)

    def smooth(self, landmarks):
        
        if len(self.landmark_buffer) == self.window_size:
            self.landmark_buffer.popleft()
        self.landmark_buffer.append(landmarks)

        if len(self.landmark_buffer) == 0:
            return landmarks

        # Compute the average landmarks
        smoothed_landmarks = np.mean(np.array(self.landmark_buffer), axis=0)
        return smoothed_landmarks
    
# Plots function =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def plot_matplotlib(values_array, width, height, points, tags):
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.plot(values_array)
    
    # Adicionar linhas verticais pontilhadas a cada 10 medidas
    for i in range(10, len(values_array), 10):
        ax.axvline(x=i, color='red', linestyle='--', linewidth=0.5)
    
    for i, point in enumerate(points):
        label = f'({point[0]}, {point[1]}, {tags[i]})'
        ax.annotate(label, (len(values_array), values_array[0][i]),
                    textcoords="offset points", xytext=(5, 5),
                    ha='center', fontsize=12, color='black')
    
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def plot_as_opencv_canvas(canvas_panel, plot_width, plot_height, values_array, points):
    color_list = [(255, 0, 0), (0, 250, 0), (0, 0, 250),
                  (0, 255, 250), (250, 0, 250), (250, 250, 0),
                  (200, 100, 200), (100, 200, 200), (200, 200, 100)]
    
    labels = [str(point) for point in points]
    
    for i in range(len(values_array) - 1):
        for j in range(len(values_array[0])):
            start_point = (i, int(plot_height * 0.9) - values_array[i][j])
            end_point = (i + 1, int(plot_height * 0.9) - values_array[i + 1][j])
            canvas_panel = cv2.line(canvas_panel, start_point, end_point, color_list[j], 1)
            if i == len(values_array) - 2:
                cv2.putText(canvas_panel, labels[j], (end_point[0] + 5, end_point[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_list[j], 1)
    
    num_ticks = 10
    tick_length = 5
    for i in range(num_ticks + 1):
        y = int(plot_height * 0.9 - (plot_height * 0.8 / num_ticks) * i)
        cv2.line(canvas_panel, (0, y), (tick_length, y), (0, 0, 0), 1)
        cv2.putText(canvas_panel, f'{int((plot_height * 0.8 / num_ticks) * i)}', 
                    (tick_length + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    
    num_ticks_x = 10
    for i in range(num_ticks_x + 1):
        x = int(plot_width * 0.9 / num_ticks_x * i)
        cv2.line(canvas_panel, (x, int(plot_height * 0.9)), (x, int(plot_height * 0.9) - tick_length), (0, 0, 0), 1)
        cv2.putText(canvas_panel, f'{int(plot_width * 0.9 / num_ticks_x * i)}', 
                    (x - 5, int(plot_height * 0.9) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    
    return canvas_panel