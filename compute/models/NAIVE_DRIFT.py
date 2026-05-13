import pandas as pd
import numpy as np

class NaiveDrift:
    def __init__(self, 
                 id_col:str, 
                 time_col:str, 
                 target:str, 
                 h:int, 
                 max_k:int=10):
        
        self.id_col = id_col
        self.time_col = time_col
        self.target = target
        self.h = h
        self.max_k = max_k
        self.best_k = None

    def fit(self, df: pd.DataFrame) -> None:
       
        df_sorted = df.sort_values(by=[self.id_col, self.time_col]).copy()
        
        best_k_per_series = {}
        for _, group in df_sorted.groupby(self.id_col):

            best_error = float('inf')
            best_k = 5

            errors = []
            for k in range(5, self.max_k + 1):
            
                y = group[self.target].values
                
                y_val = y[-self.h:]
                y_train = y[:-self.h]
                
                if len(y_train) < k + 1:
                    continue
                
                y_T = y_train[-1]
                y_T_k = y_train[-(k + 1)]
                
                drift_slope = (y_T - y_T_k) / k
                
                steps = np.arange(1, self.h + 1)
                y_hat = y_T + drift_slope * steps
                
                mae = np.mean(np.abs(y_val - y_hat))
                errors.append(mae)
                
                if mae < best_error:
                    best_error = mae
                    best_k = k

            best_k_per_series[group[self.id_col].iloc[0]] = best_k
                
        self.best_k = best_k_per_series
 
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
                   
        df_sorted = df.sort_values(by=[self.id_col, self.time_col]).copy()
        forecasts = []
        
        for serie_id, group in df_sorted.groupby(self.id_col):
            y = group[self.target].values
            last_time = group[self.time_col].max()
                            
            y_T = y[-1]
            k_used = self.best_k[serie_id] if serie_id in self.best_k else 1
            y_T_k = y[-(k_used + 1)] if len(y) > k_used + 1 else y[-1]
            drift_slope = (y_T - y_T_k) / k_used if k_used > 0 else 0
            
            steps = np.arange(1, self.h + 1)
            y_hat = y_T + drift_slope * steps
            future_times = np.arange(last_time + 1, last_time + 1 + self.h)
            
            df_forecast = pd.DataFrame({
                self.id_col: serie_id,
                self.time_col: future_times,
                'y_hat_05': np.nan,
                'y_hat_10': np.nan,
                'y_hat_50': y_hat,
                'y_hat_90': np.nan,
                'y_hat_95': np.nan,
                'model': 'NaiveDrift'
            })
            forecasts.append(df_forecast)
            
        return pd.concat(forecasts, ignore_index=True)