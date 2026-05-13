import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class SlidingWindowDataset(Dataset):
    """
    A PyTorch Custom Dataset for time series data, allowing for dynamic features,
    categorical features, and flexible encoding and prediction lengths.

    Args:
        df (pd.DataFrame): DataFrame containing time series data preprocess with columns for id, ds, target, and features.
        target (str): Name of the target variable column in the DataFrame.
        id_col (str): Name of the column in the DataFrame that identifies different time series.
        time_col (str): Name of the column in the DataFrame that contains the time information.
        cat_features (list[str]): List of categorical feature names to be used in the dataset.
        lag_features (list[str]): List of lag feature names to be used in the dataset.
        max_pred_len (int): Maximum length of the prediction (future) part of the time series.
        min_pred_len (int): Minimum length of the prediction (future) part of the time series.
        enc_len (int): Length of the encoding (historical) part of the time series.
        window_steps (int, optional): Step size for creating windows. Defaults to 1.
        train_mode (bool, optional): Whether the dataset is used for training (True) or inference (False).
        augment_series_ids (list, optional): List of series ids selected for recent-trajectory augmentation.
        augment_n_windows (int, optional): Number of most recent windows duplicated per selected series.
        augment_noise_std (float, optional): Noise std applied to augmented dynamic features and target.
        device (str, optional): Device to which the tensors will be moved ('cpu' or 'cuda'). Defaults to 'cpu'.
    """

    def __init__(self,            
                 df:pd.DataFrame,
                 target:str,
                 id_col:str,
                 time_col:str,
                 cat_features:list[str],
                 lag_features:list[str],
                 enc_len:int,
                 max_pred_len:int,
                 min_pred_len:int=None,                 
                 window_steps:int=1,
                 train_mode:bool=True,
                 augment_series_ids:list=None,
                 augment_n_windows:int=1,
                 augment_noise_std:float=0.02,
                 device:str='cpu'
                ) -> None:
        
        self.target = target
        self.id_col, self.time_col = id_col, time_col
        self.cat_features = cat_features or []
        self.lag_features = lag_features or []

        self.enc_len = enc_len
        self.max_pred_len = max_pred_len
        self.min_pred_len = min_pred_len if min_pred_len is not None else max_pred_len
        
        self.window_steps = window_steps
        self.train_mode = train_mode

        self.augment_series_ids = augment_series_ids
        self.augment_n_windows = max(int(augment_n_windows), 0)
        self.augment_noise_std = float(augment_noise_std)

        self.device = torch.device(device)
        
        self._get_windows(df)

    def _get_windows(self,
                     df:pd.DataFrame
                    ) -> None:

        windows = []
        max_win_len = self.enc_len + self.max_pred_len
        windows_cols = [self.id_col, self.time_col] + self.cat_features + self.lag_features + [self.target]

        for _, series_df in df.groupby(self.id_col):

            win_len = self.enc_len + self.max_pred_len
            win_num = len(series_df) - win_len + 1 - (self.min_pred_len if self.train_mode else 0)
            padding_len = max_win_len - win_len

            if win_num <= 0:
                continue

            win_idx = np.arange(win_len)[None, :] + np.arange(win_num)[:, None]

            if self.train_mode:
                win_steps = np.arange(0, win_num, self.window_steps)
                win_steps = win_steps[::-1]
            else:
                win_steps = np.array([-1])
                    
            grp_windows = series_df[windows_cols].values[win_idx[win_steps]]

            if padding_len > 0:
                grp_windows = np.pad(grp_windows, ((0, 0), (padding_len, 0), (0, 0)), mode='constant', constant_values=0)
            
            windows.append(grp_windows)
        
        windows = np.concatenate(windows, axis=0)

        if self.train_mode and self.augment_series_ids is not None:
            windows = self._augment_windows(windows, windows_cols)

        self.metadata = {'id': windows[:, 0, windows_cols.index(self.id_col)],
                         'y_times': windows[:, -self.max_pred_len:, windows_cols.index(self.time_col)]}
        
        self.model_inputs = {'x_dyn': torch.tensor(windows[:, :self.enc_len, [windows_cols.index(feat) for feat in self.lag_features + [self.target]]], dtype=torch.float32).to(self.device),
                             'x_cat': torch.tensor(windows[:, 0, [windows_cols.index(feat) for feat in self.cat_features]],
                                                   dtype=torch.long).to(self.device) if self.cat_features else None,
                             'y': torch.tensor(windows[:, -self.max_pred_len:, windows_cols.index(self.target)], dtype=torch.float32).to(self.device)}
        
    def __len__(self) -> int:
        return len(self.model_inputs['x_dyn'])
    
    def __getitem__(self, 
                    idx: int
                   ) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:

        model_inputs_item = {key: tensor[idx] for key, tensor in self.model_inputs.items() if tensor is not None}
        
        metadata_item = {key: array[idx] for key, array in self.metadata.items()}

        return model_inputs_item, metadata_item
    
    def _augment_windows(self,
                         windows:np.ndarray,
                         windows_cols:list[str]
                        ) -> np.ndarray:
        
        id_idx = windows_cols.index(self.id_col)
        window_ids = windows[:, 0, id_idx]
        mask = np.zeros(len(windows), dtype=bool)
        selected_ids = np.unique(window_ids[np.isin(window_ids, self.augment_series_ids)])
        for series_id in selected_ids:
            series_idx = np.flatnonzero(window_ids == series_id)
            if len(series_idx) > 0:
                n_recent = min(self.augment_n_windows, len(series_idx))
                mask[series_idx[:n_recent]] = True
        wins_samples = windows[mask]

        if len(wins_samples) == 0 or self.augment_n_windows <= 0:
            return windows
        
        augmented_list = [windows]
        noise_idx = [windows_cols.index(feat) for feat in self.lag_features + [self.target]]

        wins_augmented = wins_samples.copy()
        noise = np.random.normal(
            0,
            self.augment_noise_std,
            size=(wins_augmented.shape[0], wins_augmented.shape[1], len(noise_idx)),
        )
        wins_augmented[:, :, noise_idx] += noise
        wins_augmented[:, :, id_idx] += 1000
        augmented_list.append(wins_augmented)

        return np.concatenate(augmented_list, axis=0)
