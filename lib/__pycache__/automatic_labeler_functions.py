import stumpy, os, pandas as pd, numpy as np, ast, re

import manual_labeler_functions as man_lab_fun 

#from stumpy import config
#config.STUMPY_EXCL_ZONE_DENOM = 1

# Class to manager the searches
class Comparing:
    def __init__(self, Q_df, T_df, distance_threshold):
        self.Q_df = Q_df # referencia
        self.T_df = T_df # serie
        self.matches_idxs = [] # index das subséries mais próximas da referencia encontradas pela função match
        self.filter_matches_idxs = [] # index selecionados por similaridade entre várias medidas
        self.measure_name = None # nome da medida
        self.distance_threshold = distance_threshold

    def calc_matches(self):
        self.matches_idxs = stumpy.match(self.Q_df, self.T_df, max_distance=lambda D: max(np.mean(D) - self.distance_threshold * np.std(D), np.min(D)))

def CREATE_FILES_INDEX(FILE_LOCATION_TREE_DLOCAL, VD_INFO_FILE_NAME = 'VD_INFO.CSV', VD_LABEL_FILE_NAME = 'VD_LABELED_LO.CSV'):
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
        
    return INDEX_REF_DT

def get_euclidean_distance(target, matrix):
    for subarray in matrix:
        if target in subarray:
            return subarray[0]
    return None

def label_current_series(current_path_location, RESUME_DT, selected_measures_in_frame_interval, dict_label_parameters, seed_name, LABELED_FILE_PATH='VD_LABELED_L0.CSV', distance_threshold=2, frame_threshold=3):
    VD_MEASURE_DT = pd.read_csv(current_path_location)
    VD_MEASURE_DT.drop(columns=['Unnamed: 0'], inplace=True)

    T_df = VD_MEASURE_DT[dict_label_parameters['reference_measures']]

    # Aply Stumpy functions
    object_list = []
    temp_row = pd.DataFrame()

    for step in range(0, len(selected_measures_in_frame_interval.columns)):
        comp_object = Comparing(selected_measures_in_frame_interval[dict_label_parameters['reference_measures'][step]], T_df[dict_label_parameters['reference_measures'][step]], distance_threshold)
        comp_object.calc_matches()
        
        #print("Matches_idxs:", comp_object.matches_idxs)
        comp_object.measure_name = dict_label_parameters['reference_measures'][step]
        object_list.append(comp_object)
        
        # contabiliza resultado
        temp_row.at[0, dict_label_parameters['reference_measures'][step]] = int(len(comp_object.matches_idxs))
        
    # Aply filter of matching
    all_index = []
    for c in object_list:  
        all_index.append(c.matches_idxs[:, 1])
    
   
    # filtra por coincidência a partir de um um treshold de distancia entre a posição dos indexs
    aux = all_index.copy()
    filter_index = find_all_matches(aux, frame_threshold)
    
    filter_index_list=list(filter_index[0])

    # Corrigir os index das subséries pelo index do frame original(frame_seq)
    filter_index_begin = []
    for idx_tuple in filter_index_list:
        filter_index_begin.append(idx_tuple)


    idxs_match_frame_seq = []
    for idx in filter_index_begin:
        idx_frame_seq = VD_MEASURE_DT.loc[idx, 'frame_seq']
        for c in object_list:
            ed = get_euclidean_distance(idx, c.matches_idxs)
            if ed != None: break
        idxs_match_frame_seq.append([idx_frame_seq, ed])
            
            
    # Test if the Labeled File was already created
    test = os.path.exists((os.path.join(os.path.dirname(current_path_location), LABELED_FILE_PATH)))
    VD_LABEL_PATH = (os.path.join(os.path.dirname(current_path_location), LABELED_FILE_PATH))
    
    if test:
        print('Reading Label File...') 
        VD_LABELED_DT = pd.read_csv(VD_LABEL_PATH)
        VD_LABELED_DT.drop(columns=['Unnamed: 0'], inplace=True)
        VD_LABELED_DT = VD_LABELED_DT.set_index(pd.Index(VD_LABELED_DT['frame_seq']))
    else:
        print('Creating Label File...')
        
        # First Initiate the labels = 0 means NO Label
        VD_LABELED_DT = VD_MEASURE_DT.copy()
        VD_LABELED_DT['label_measures'] = str({})
        VD_LABELED_DT = VD_LABELED_DT.set_index(pd.Index(VD_LABELED_DT['frame_seq']))
    
    temp_row['final'] = (len(filter_index[0]))

    gap_ocurrences = 0

    # Adds information to label the frames.
    for label_idx in idxs_match_frame_seq:
        init_lab = label_idx[0]
        endd_lab = init_lab+len(selected_measures_in_frame_interval)-1
        e_distace = label_idx[1]
        FRAMES_DT = VD_LABELED_DT.query(f'frame_seq >= {init_lab} & frame_seq <= {endd_lab}')
        
        # if there is not a descontinuity in the interval of frames
        if not FRAMES_DT['gap'].any()==1 and endd_lab in VD_LABELED_DT.index:
            # in cases that the missing frames are in the end of interval
            VD_LABELED_DT = UPDATE_LABEL_DF(init_lab, endd_lab, dict_label_parameters['label_name'], dict_label_parameters['reference_measures'], VD_LABELED_DT, seed_name, e_distace)
        else:
            gap_ocurrences += 1

    temp_row['final'] -= gap_ocurrences
    RESUME_DT = pd.concat([RESUME_DT, temp_row], axis=0)
    
    # Save CSV file
    print('Saving VD_LABELED_L0...')
    VD_LABELED_DT.drop(columns=['frame_seq'], inplace=True)
    VD_LABELED_DT.reset_index(inplace=True)
    VD_LABELED_DT.to_csv(VD_LABEL_PATH)

    return RESUME_DT

def UPDATE_LABEL_DF (init_lab, endd_lab, label_name_in, label_measur_in, data_frame_in, seed_name, matches_idxs):
    
    # Check if END is Greater than Length
    for index_x in range(init_lab, endd_lab+1):
        idx_retur_str = data_frame_in['label_measures'][index_x]
        dicct_current = ast.literal_eval(idx_retur_str)

        match = re.search(r'VD_R_(\d+)', seed_name)
        video_num = int(match.group(1)) if match else seed_name

        # Insert Updating DICT
        dicct_current.update ({label_name_in: (label_measur_in, matches_idxs, video_num)})
        
        # Put Dict into the Current DATA FRAME
        data_frame_in.loc[index_x, 'label_measures'] = str(dicct_current)
    
    return data_frame_in

def find_close_values(idxs, threshold):
    close_values = []

    # Comparação entre cada par de listas
    for i in range(len(idxs)):
        for j in range(i+1, len(idxs)):
            list1 = idxs[i]
            list2 = idxs[j]

            # Comparação de cada elemento entre as duas listas
            for num1 in list1:
                for num2 in list2:

                    # Se a diferença entre os valores for menor ou igual ao threshold, considere-os próximos
                    if abs(num1 - num2) <= threshold:
                        close_values.append(min(num1,num2))
    close_values = set(close_values)
    return close_values

def find_all_matches(list_of_index, threshold):
    n = len(list_of_index)
    list_aux = []
    
    if n <= 1:  
        return list_of_index
    else: 
        # Seleciona o primeiro e segundo elemento da lista para a busca de similaridade
        list_aux.append(list_of_index.pop(0))
        list_aux.append(list_of_index.pop(0))

        result = find_close_values(list_aux, threshold)
        list_of_index.insert(0, result)
        
        return find_all_matches(list_of_index, threshold)
