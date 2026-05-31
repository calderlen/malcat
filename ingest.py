# interate through the light curve files
# these are flat in the output bundles
# each has a raw2 and dat2 -- raw2 can likely be neglected for now b/c the transformer may learn these features on its own -- dont want to overcode
# so ingest the dat2 files

import os
from time import time
import pandas as pd
import torch
from torch.utils.data import Dataset


def concatenate_folders(folder_list, output_folder, f_ext='.dat2'):

    os.makedirs(output_folder, exist_ok=True)

    # if the output folder is empty, then concatenate the folders. if not empty, then assume the folders have already been concatenated and do nothing
    if not os.listdir(output_folder):
        for folder in folder_list:
            for file in os.listdir(folder):
                if file.endswith(f_ext):
                    source_file = os.path.join(folder, file)
                    destination_file = os.path.join(output_folder, file)
                    os.symlink(source_file, destination_file)

def normalize_folder_list(lc_folders):
    if isinstance(lc_folders, (str, os.PathLike)):
        return [os.fspath(lc_folders)]
    return [os.fspath(folder) for folder in lc_folders]

def collect_lc_files(lc_folders, f_ext='.dat2'):
    files = []
    for folder in normalize_folder_list(lc_folders):
        for file in os.listdir(folder):
            if file.endswith(f_ext):
                files.append(os.path.join(folder, file))
    return sorted(files)

def load_lc(path):
    cols = ['jd', 'mag', 'mag_err', 'good', 'camera', 'band', 'saturated', 'cam/field']
    df = pd.read_csv(path, sep=r'\s+', header=None, names=cols)
    return df

def drop_bad_data(df):
    #TODO: ensure this is the right convention
    df = df[df['good'] == 1]
    df = df[df['saturated'] == 0]
    return df

def relative_time(df):
    # convert jd to relative jd
    df['delta_jd'] = df['jd'] - df['jd'].min()
    return df

def correct_band_offsets(df, target_col='mag', output_col='mag_band_corr'):
    # compute median of each band, then subtract the median of each band from the magnitudes in that band to get the corrected magnitudes
    df_band_med = df.groupby('band')[target_col].transform('median')
    df[output_col] = df[target_col] - df_band_med
    return df

def correct_camera_offsets(df, target_col='mag_band_corr', output_col='mag_cam_corr'):
    # compute median of each camera, then subtract the median of each camera from the magnitudes in that camera to get the corrected magnitudes
    df_cam_med = df.groupby('camera')[target_col].transform('median')
    df[output_col] = df[target_col] - df_cam_med
    return df

def relative_mag(df, target_col='mag_cam_corr'):
    # this function may not be necessary
    df['delta_mag'] = df[target_col] - df[target_col].median()
    return df

def preprocess_lc(path):
    df = load_lc(path)
    df = drop_bad_data(df)
    df = relative_time(df)
    df = correct_band_offsets(df)
    df = correct_camera_offsets(df)
    df = relative_mag(df)

    return df


class LightCurveDataset(Dataset):
    
    def __init__(self, lc_folders, f_ext='.dat2'):
        # setup the dataset
        self.lc_folders = normalize_folder_list(lc_folders)
        self.files = collect_lc_files(self.lc_folders, f_ext=f_ext)
        self.feature_cols = ['delta_jd', 'delta_mag']

    def __len__(self):
        # return number of examples
        return len(self.files)
    
    def __getitem__(self, idx):
        # return one example
        path = self.files[idx]
        df = preprocess_lc(path)


        x = torch.tensor(df[self.feature_cols].values, dtype=torch.float32)

        return {
            'lc': x,
            'filename': os.path.basename(path),
            'path': path,
        }

def collate_lcs(batch):
    batch_size = len(batch)
    max_len = max(item['lc'].shape[0] for item in batch)
    feature_dim = batch[0]['lc'].shape[1]

    padded_lc = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32)
    padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        lc = item['lc']
        lc_len = lc.shape[0]
        padded_lc[i, :lc_len] = lc
        padding_mask[i, :lc_len] = False

    return {
        'lc': padded_lc,
        'mask': padding_mask,
        'filename': [item['filename'] for item in batch],
        'path': [item['path'] for item in batch],
    }
