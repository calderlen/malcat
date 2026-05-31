# interate through the light curve files
# these are flat in the output bundles
# each has a raw2 and dat2 -- raw2 can likely be neglected for now b/c the transformer may learn these features on its own -- dont want to overcode
# so ingest the dat2 files

import os
from time import time
import pandas as pd
from torch.utils.data import Dataset, DataLoader

def load_lc(path):
    cols = ['jd', 'mag', 'mag_err', 'good', 'camera', 'band', 'saturated', 'cam/field']
    df = pd.read_csv(path, sep=r'\s+', header=None, names=cols)
    return df

def drop_bad_data(df):
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

def numerical_lc(df, time_col='delta_jd', mag_col='delta_mag', mag_err_col='mag_err'):
    #TODO: for now ignoring mag_err outright since i believe every datapoint has the same 0.1 mag error bar; OR get the better data from skypatrol. decide on whether this is final
    df_num = df[[time_col, mag_col]].copy()
    return df_num

def normalize_lc_length(df, target_length, time_col='delta_jd', mag_col='delta_mag', mask_col='padding_mask'):
    # pad or truncate the light curve to the target length. if truncating, find the length of the light curve. then from those data points, drop random sample of datapoints from the light curve until the target length is reached. if padding, add rows of zeros to be masked later.

    lc_len = len(df)
    
    if lc_len > target_length:
        df = df.sample(n=target_length, random_state=7).sort_values(time_col).reset_index(drop=True)
        df[mask_col] = False

    elif lc_len < target_length:
        df[mask_col] = False
        num_pad = target_length - lc_len

        padding = pd.DataFrame({
            time_col: [0.] * num_pad, 
            mag_col: [0.] * num_pad, 
            mask_col: [True] * num_pad})
        
        df = pd.concat([df, padding], ignore_index=True)
        
    else:
        df[mask_col] = False
        
    return df


class LightCurveDataset(Dataset):
    def __init__(self, lc_folder, f_ext='.dat2'):
        self.lc_folder = lc_folder
        self.files = [f for f in os.listdir(lc_folder) if f.endswith(f_ext)]

        def __len__(self):
            return len(self.files)
        




            return df[self.cols]


# now create tensors for the transformer model