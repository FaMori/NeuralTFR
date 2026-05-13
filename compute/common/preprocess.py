import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class TimeSeriesPreprocess():
    """
    A class for preprocessing time series data for forecasting tasks.
    It includes methods for fitting scalers, transforming data, splitting into training and validation sets,
    and preparing data for predictions.
    Attributes:
        target (str): Name of the target variable.
        time_col (str): Name of the time column.
        id_col (str): Name of the identifier column.
        cat_features (list[str]): List of categorical features.
        lags_features (list[int]): List of lag periods to create lag features.
        apply_log (bool): Whether to apply log transformation to the target variable.
        scaler (str): Type of scaler to use for scaling features and target variable.
            - 'Standard': StandardScaler
            - 'MinMax': MinMaxScaler
            - 'Robust': RobustScaler
    """

    def __init__(self,
                 target:str='TFR',
                 time_col:str='year',
                 id_col:str='id',
                 cat_features:list[str]=None,
                 lags_features:list[int]=None,
                 apply_log:bool=False,
                 scaler:str='Standard'
                 ) -> None:
        
        self.target = target
        self.time_col, self.id_col = time_col, id_col

        self.apply_log = apply_log
        
        available_scalers = {'Standard':StandardScaler,'MinMax':MinMaxScaler,'Robust':RobustScaler}
        self.scaler = available_scalers[scaler]()
        self.params_scaler = None

        self.cat_features = cat_features or []
        self.n_cat_features = []
        self.cat_mappings = {}

        self.lags_features = lags_features or []
        self.lag_feats = []

    def fit(self,
            df:pd.DataFrame
            ) -> 'TimeSeriesPreprocess':
        
        fit_df = df.copy()
        
        fit_df = fit_df[[self.id_col,self.time_col,self.target] + self.cat_features]
        fit_df['y'] = fit_df[self.target]

        if self.apply_log:
            fit_df['y'] = np.log(fit_df['y'])

        self.params_scaler = self.scaler.fit(fit_df[['y']])
        
        if self.cat_features:
            fit_df = self._process_categories(fit_df, set_map=True)
            
        return self 
    
    def transform(self,
                  df:pd.DataFrame,
                  h:int=0
                 ) -> pd.DataFrame:
        
        prep_df = df.copy()

        prep_df = prep_df[[self.id_col,self.time_col,self.target] + self.cat_features]
        prep_df['y'] = prep_df[self.target]

        if self.apply_log:
            prep_df['y'] = np.log(prep_df['y'])
        
        prep_df['y'] = self.params_scaler.transform(prep_df[['y']])
        
        if self.cat_features:
            prep_df = self._process_categories(prep_df, set_map=False)

        if self.lags_features:
            prep_df = self._add_lags(prep_df)
        
        if h > 0:
            prep_df = self._extend_df(prep_df, extend_h=h)

        prep_df = prep_df[[self.id_col, self.time_col, self.target, 'y'] + self.lag_feats + self.cat_features]
                
        return prep_df
    
    def inverse_transform(self,
                          df:pd.DataFrame,
                          y_hat_cols:list[str]=['y_hat']
                         ) -> pd.DataFrame:
        
        inv_df = df.copy()
        
        for col in y_hat_cols:
            inv_df[col] = self.params_scaler.inverse_transform(inv_df[[col]])
        
        if self.apply_log:
            inv_df[y_hat_cols] = np.exp(inv_df[y_hat_cols])

        return inv_df
    
    def split_train_valid(self,
                          df:pd.DataFrame,
                          h:int,
                          enc_len:int
                         ) -> tuple[pd.DataFrame,pd.DataFrame]:
        
        self.valid_split = df[self.time_col].max() - h
        train_df = df[df[self.time_col] <= (self.valid_split)].copy() 

        overlap_val = self.valid_split - enc_len
        valid_df = df[df[self.time_col] > (overlap_val)].copy()

        return train_df, valid_df
    
    def _add_lags(self,
                  df:pd.DataFrame
                 ) -> pd.DataFrame:
        
        df.sort_values(by=[self.id_col, self.time_col], inplace=True)
        for lag in self.lags_features:

            new_col = f"y_lag{lag}"
            df[new_col] = df.groupby(self.id_col)['y'].shift(lag)
            df[new_col] = df.groupby(self.id_col)[new_col].bfill()
        
            if new_col not in self.lag_feats:
                self.lag_feats.append(new_col)

        return df
        
    def _extend_df(self,
                   df:pd.DataFrame,
                   extend_h:int
                  ) -> pd.DataFrame:
        
        ext_groups_df = []
        for serie_id, group_df in df.groupby(self.id_col):

            h0_ds = group_df[self.time_col].max() + 1
            ext_groups_df.append(pd.DataFrame({self.id_col: serie_id,
                                 self.time_col: np.arange(h0_ds, h0_ds + extend_h)}))

        ext_df = pd.concat(ext_groups_df, ignore_index=True)
        
        df = pd.concat([df, ext_df], ignore_index=True).sort_values(by=[self.id_col,self.time_col]).reset_index(drop=True)

        return df
       
    def _process_categories(self,
                            df:pd.DataFrame,
                            set_map:bool=False
                           ) -> pd.DataFrame:
        
        if set_map:
            self.n_cat_features = [] 
            for col in self.cat_features:
                categories = df[col].unique()
                self.cat_mappings[col] = {cat: i + 1 for i, cat in enumerate(categories)}
                self.n_cat_features.append(len(categories) + 1)
        
        for col in self.cat_features:
            df[col] = df[col].map(self.cat_mappings[col]).fillna(0).astype(int)

        return df
