import os
from sklearn.model_selection import StratifiedGroupKFold
from utilites import seed_everything, prepare_data, load_config, show_image
import shutil
import json 
import argparse
import numpy as np

def split_files_for_cross_validation(cfg):
    seed_everything(cfg['seed'])
    
    datadir = cfg['datadir']
    data_list = prepare_data(datadir)

    #define dir where to copy data
    copydata_dir = '/mnt/atlas/data_KPIs/data/KPIs24_Training_Data/Task1_patch_level/split/'
    # os.makedirs(copydata_dir, exist_ok=True)
    
    #join id and class
    for i in range(len(data_list)):
        data_list[i]['img_stratify'] = data_list[i]['case_id'] + '_' + data_list[i]['case_class']
        
    #split data_list for cross validation - grouped stratify split based on type
    sgkf = StratifiedGroupKFold(n_splits=cfg['nfolds'], shuffle=True, random_state=cfg['seed'])
    
    for index_fold, (train_index, val_index) in enumerate(sgkf.split(data_list, [d['case_class'] for d in data_list], [d['img_stratify'] for d in data_list])):
        
        #copy valition data to a folder with the fold number+
        fold_dir = os.path.join(copydata_dir, 'fold_' + str(index_fold))
        # os.makedirs(fold_dir, exist_ok=True)
        for i in val_index:
            #get case id from the image id
            case_id = data_list[i]['case_id']
            #get class id
            class_id = data_list[i]['case_class']
            #create a folder for the case id
            case_dir = os.path.join(fold_dir, class_id, case_id, )
            os.makedirs(case_dir, exist_ok=True)
            shutil.copy(data_list[i]['img'], case_dir)
            print('copying', data_list[i]['img'], 'to', case_dir)

        #save the fold data to a json file
        fold_data = {
            'data': [data_list[i] for i in val_index],
            'class': list(np.unique([data_list[i]['case_class'] for i in val_index])),
            'class_counts': {class_id: len([data_list[i]['case_class'] for i in val_index if data_list[i]['case_class'] == class_id]) for class_id in np.unique([data_list[i]['case_class'] for i in val_index])},
            'id_per_class_counts': {class_id: len(set([data_list[i]['case_id'] for i in val_index if data_list[i]['case_class'] == class_id])) for class_id in np.unique([data_list[i]['case_class'] for i in val_index])},
            'fold': index_fold,
            'split': 'val',
            'split_pathes_size': len(val_index),
            'total_pathes_size': len(data_list),
            'split_id_size': len(set([data_list[i]['case_id'] for i in val_index])),
            'seed': cfg['seed']
        }
        print('fold', index_fold, ':' , fold_data['class_counts'])
        print('fold', index_fold, ':' , fold_data['id_per_class_counts'])
        # #save json file
        fold_json_path = os.path.join(copydata_dir, f'fold_{index_fold}.json')
        with open(fold_json_path, 'w') as f:
            json.dump(fold_data, f)
    return fold_data
    
def main(cfg):
    cfg = load_config(cfg)
    split_files_for_cross_validation(cfg)
    
if __name__ == "__main__":
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", help="configuration file", default="/home/benito/script/NephroBIT/KPIs24/config_train_swinUNETR.yaml")
    # args = parser.parse_args()
    # cfg = args.config
    cfg = '/home/benito/script/NephroBIT/KPIs24/config_train_swinUNETR.yaml'
    main(cfg)