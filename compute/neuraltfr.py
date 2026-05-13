import torch
import random
import pickle
import os
import threading

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from common.preprocess import TimeSeriesPreprocess
from common.dataset import SlidingWindowDataset
from common.losses import MQLoss
from models.ENC_DEC_GRU import ENC_DEC_GRU

class NeuralTFR():
    """
    NeuralTFR class for Total Fertility Rate (TFR) forecasting using deep learning models.
    This class provides methods for training and predicting.
    Construct Arguments:
        model_name (str): Name of the model (if is save go to folder path runs/models/model_name).
        model_arq (str): Model architecture to be used (Availables: ENC_DEC_GRU).
            - ENC_DEC_GRU: Encoder GRU and Decoder GRU.
        enc_len (int): Encoder single integer for all encoder lengths.
        pred_len (list[int]|int): List of prediction lengths or single integer for all prediction lengths.
        window_steps (int): Number of steps for the sliding window.
        cat_features (list[str]): List of categorical features.
        lags_features (list[int]): List of target lags to be used as features.
        preprocess_config (dict): Configuration for preprocessing.
        model_config (dict): Configuration for the model architecture.
        random_hyperparams (dict): Dictionary of hyperparameters to be randomly sampled for each model in the ensemble.
        augment_id_tfr_le (float): Raw TFR threshold used to auto-select ids whose recent training trajectory will be augmented.
        augment_recent_n_windows (int): Number of most recent training windows duplicated per selected id.
        augment_noise_std (float): Noise std used when perturbing augmented windows.
        loss_function (MQLoss): Loss function to be used for training. Defaults to MQLoss with quantiles [0.05,0.1,0.5,0.9,0.95].
        n_models (int): Number of models to train in the ensemble.
        save_model (bool): Whether to save the trained model or not.
        seed (int): Random seed for reproducibility.
        device (str): Device to be used for training ('cpu' or 'cuda').
        torch_threads (int): Number of threads for PyTorch on CPU.
        log (bool): Whether to log information or not.
    """

    def __init__(self,
                 model_name:str='NeuralTFR',
                 model_arq:str='ENC_DEC_GRU',
                 enc_len:int=20,
                 pred_len:list[int]=[10,15],
                 window_steps:int=1,
                 cat_features:list[str]=None,
                 lags_features:list[int]=None,
                 preprocess_config:dict={'target':'TFR','time_col':'year','id_col':'id',
                                         'apply_log':False, 'scaler':'Standard'},
                 model_config:dict={'enc_hidden_size':8,'dec_hidden_size':8,'dim_embedding':None},
                 random_hyperparams:dict=None,
                 augment_tfr_level:float=None,
                 augment_n_windows:int=1,
                 augment_noise_std:float=0.02,
                 loss_function:MQLoss=MQLoss(quantiles=[0.05,0.1,0.5,0.9,0.95]),
                 n_models:int=1,
                 save_model:bool=False,
                 seed:int=1234,
                 device:str='cpu',
                 torch_threads:int=2,
                 log:bool=True,
                 deterministic:bool=False
                ) -> None:
        
        self.model_name, self.model_path = model_name, os.path.join('runs', 'models', model_name)
        self.num_models, self.model_arq  = n_models, model_arq
        self.model_config, self.save_model = model_config, save_model

        self.device = torch.device(device)
        if device == 'cpu':
            torch_threads = 1 if n_models > 1 else torch_threads
            torch.set_num_threads(torch_threads)

        self.seed = seed
        self.deterministic = deterministic

        self.enc_len = enc_len
        self.max_pred_len, self.min_pred_len = max(pred_len), min(pred_len)
        self.window_steps = window_steps

        self.ts_preprocess = TimeSeriesPreprocess(**preprocess_config, cat_features=cat_features, lags_features=lags_features)
        self.augment_tfr_level = augment_tfr_level
        self.augment_n_windows = max(int(augment_n_windows), 0)
        self.augment_noise_std = float(augment_noise_std)
        self.loss_function = loss_function
        self.log = log

        self.random_hyperparams = random_hyperparams or {}
        self.ensemble_configs:list[dict] = []
        self.models_states:list[dict] = []

    def fit(self,
            df:pd.DataFrame,
            epochs:int=100,
            max_steps:int=-1,
            batch_size:int=32,
            optimizer_config:dict={'lr':1e-3,'weight_decay':0.0,'type_scheduler':'Step','step_size':10,'gamma':0.98},
            valid_epochs:int=0,
            stop_patience:int=2,
            reset_weights:bool=False,
            n_jobs:int=1,    
           ) -> 'NeuralTFR':
                
        self.training_summary = {'last_epochs':[], 'last_steps':[]}
        self.training_metrics = None
        self._set_seed(self.seed)
        
        initial_states = []
        if reset_weights:
            initial_states = [None] * self.num_models 
            self.models_states = [] 
        elif len(self.models_states) == self.num_models:
            initial_states = self.models_states 
            self.models_states = []
        else:
            initial_states = [None] * self.num_models
            self.models_states = []

        if valid_epochs > 0:
            self.train_df, self.valid_df = self.ts_preprocess.split_train_valid(df=df,h=self.max_pred_len,enc_len=self.enc_len)
        else:
            self.train_df, self.valid_df = df.copy(), None

        self.prep_train_df = self.ts_preprocess.fit(df=self.train_df).transform(df=self.train_df,h=self.max_pred_len)
        self.prep_valid_df = self.ts_preprocess.transform(df=self.valid_df,h=self.max_pred_len) if valid_epochs > 0 else None
        augment_series_ids = self._augment_series_ids(self.train_df)

        dataset_params = {'target': 'y', 'id_col': self.ts_preprocess.id_col, 'time_col': self.ts_preprocess.time_col,
                          'cat_features': self.ts_preprocess.cat_features, 'lag_features': self.ts_preprocess.lag_feats,
                          'enc_len': self.enc_len, 'max_pred_len': self.max_pred_len, 'min_pred_len': self.min_pred_len, 'device': self.device.type}
    
        self.train_dataset = SlidingWindowDataset(df=self.prep_train_df,window_steps=self.window_steps,train_mode=True, 
                                                  augment_series_ids=augment_series_ids,
                                                  augment_n_windows=self.augment_n_windows,
                                                  augment_noise_std=self.augment_noise_std,
                                                  **dataset_params)

        self.valid_dataset = SlidingWindowDataset(df=self.prep_valid_df, train_mode=True, **dataset_params) if valid_epochs > 0 else None

        if self.log:
            n_series = self.train_df[self.ts_preprocess.id_col].nunique()
            n_train_windows = len(self.train_dataset)
            n_valid_windows = len(self.valid_dataset) if valid_epochs > 0 else 0
            
            print("=" * 55)
            print("NeuralTFR Training")
            print("-" * 55)
            print(f"Ensemble size      : {self.num_models} models ({self.model_arq})")
            print(f"Time series count  : {n_series}")
            print(f"Train samples      : {n_train_windows} windows")
            if valid_epochs > 0:
                print(f"Valid samples      : {n_valid_windows} windows")
            print("=" * 55)

        ncat_embedding = self.ts_preprocess.n_cat_features if self.ts_preprocess.cat_features else None
        self.model_sizes = {'h':self.max_pred_len,'output_size':self.loss_function.output_size,
                            'enc_feat_size':1 + len(self.ts_preprocess.lag_feats),'ncat_embedding':ncat_embedding}
                
        self.available_models = {'ENC_DEC_GRU': ENC_DEC_GRU}
        seeds = [int(self.seed) + i for i in range(self.num_models)]
        if len(self.ensemble_configs) != self.num_models:
            self.ensemble_configs = []
            for i in range(self.num_models):
                self._set_seed(seeds[i]) 
                
                sampled_hyperparams = self._sample_hyperparams()
                job_model_config = self.model_config.copy()
                job_model_config.update(sampled_hyperparams)
                
                self.ensemble_configs.append({
                    'config': job_model_config,
                    'sampled': sampled_hyperparams
                })
        
        tqdm.set_lock(threading.RLock())
        def _train_model_job(member_idx:int) -> dict:
          
            job_seed = seeds[member_idx]
            self._set_seed(job_seed)

            job_config_data = self.ensemble_configs[member_idx]
            job_model_config = job_config_data['config']
            sampled_hyperparams = job_config_data['sampled']

            model_i = self.available_models[self.model_arq](
                **job_model_config, 
                **self.model_sizes
            ).to(self.device)

            initial_state_dict = initial_states[member_idx]
            if initial_state_dict is not None:
                model_i.load_state_dict(initial_state_dict)

            worker_position = member_idx % n_jobs
            current_pbar = None
            if self.log:
                current_pbar = tqdm(
                    total=epochs, 
                    position=worker_position, 
                    desc=f"Model {member_idx+1}", 
                    leave=False, 
                    unit=" Epochs"
                )

            from common.trainer import Trainer
            trainer_i = Trainer(model=model_i,loss_function=self.loss_function, 
                                optimizer_config=optimizer_config,log=self.log,
                                pbar=current_pbar)

            trained_model_i = trainer_i(self.train_dataset,self.valid_dataset, 
                                        batch_size,epochs,max_steps,
                                        valid_epochs,stop_patience)

            return {'state_dict': trained_model_i.state_dict(),'train_losses': trainer_i.train_losses,'valid_losses': trainer_i.valid_losses,
                    'last_epoch': trainer_i.last_epoch,'last_step': trainer_i.last_step,'sampled_hyperparams': sampled_hyperparams}
        
        n_jobs = n_jobs or self.num_models
        parallel_backend = "loky" if self.deterministic and n_jobs > 1 and self.device.type == 'cpu' else "threading"
        trained_models = Parallel(n_jobs=n_jobs, backend=parallel_backend)(
                delayed(_train_model_job)(i) for i in range(self.num_models)
            )
        
        logs_df = []
        self.models_states = []
        for idx, model in enumerate(trained_models):

            self.models_states.append(model['state_dict'])
            self.training_summary['last_epochs'].append(model['last_epoch'])
            self.training_summary['last_steps'].append(model['last_step'])

            df_train = pd.DataFrame.from_dict(model['train_losses'], orient='index', columns=['train_losses'])
            df_valid = pd.DataFrame.from_dict(model['valid_losses'], orient='index', columns=['valid_losses'])

            log_df = df_train.join(df_valid, how='outer')
            log_df.index.name = 'epoch'
            log_df['ensemble'] = idx + 1
            for hparam_key, hparam_value in model['sampled_hyperparams'].items():
                if isinstance(hparam_value, list):
                    log_df[hparam_key] = str(hparam_value)
                else:
                    log_df[hparam_key] = hparam_value

            logs_df.append(log_df)
        
        self.training_metrics = pd.concat(logs_df).reset_index()

        if self.save_model:
            model_save_path = os.path.join(self.model_path, 'model', 'model.pkl')
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

            with open(model_save_path, 'wb') as model:
                pickle.dump(self, model)

        return self
    
    def predict(self,
                df:pd.DataFrame
                ) -> pd.DataFrame:
        
        prep_df = self.ts_preprocess.transform(df,h=self.max_pred_len)

        dataset = SlidingWindowDataset(df=prep_df,target='y',
                                       id_col=self.ts_preprocess.id_col,time_col=self.ts_preprocess.time_col,
                                       cat_features=self.ts_preprocess.cat_features, lag_features=self.ts_preprocess.lag_feats,
                                       enc_len=self.enc_len, max_pred_len=self.max_pred_len, device=self.device.type, train_mode=False)
        
        all_predicts_list = []
        for i in range(self.num_models):
            
            model_config = self.ensemble_configs[i]['config']
        
            state_dict = self.models_states[i]
            model_i = self.available_models[self.model_arq](
                **model_config, 
                **self.model_sizes).to(self.device)

            model_i.load_state_dict(state_dict, strict=False) 
            model_i.eval()

            with torch.no_grad():
                if hasattr(model_i, 'tf_config'):
                    model_i.tf_rate = 0.0
                y_hat = model_i(dataset.model_inputs)

            all_predicts_list.append(y_hat.cpu().numpy())

        y_hat_np = np.median(all_predicts_list, axis=0)
        predict_df = pd.DataFrame(y_hat_np.reshape(-1, y_hat_np.shape[-1]),columns=self.loss_function.output_cols)
        predict_df[self.ts_preprocess.id_col] = dataset.metadata[self.ts_preprocess.id_col].repeat(self.max_pred_len)
        predict_df[self.ts_preprocess.time_col] = dataset.metadata['y_times'].reshape(-1)

        predict_df = self.ts_preprocess.inverse_transform(predict_df,y_hat_cols=self.loss_function.output_cols)
        predict_df = predict_df[[self.ts_preprocess.id_col,
                                 self.ts_preprocess.time_col] + self.loss_function.output_cols].astype({self.ts_preprocess.id_col:int,self.ts_preprocess.time_col:int})
        predict_df['model'] = self.model_name

        return predict_df.reset_index(drop=True)

    def _augment_series_ids(self,
                            df:pd.DataFrame
                           ) -> list[int] | None:

        if self.augment_tfr_level is None:
            return None

        target_col = self.ts_preprocess.target
        id_col = self.ts_preprocess.id_col
        augment_series_ids = (
            df.loc[df[target_col] <= self.augment_tfr_level, id_col]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        if not augment_series_ids:
            return None

        return sorted(augment_series_ids)
    
    def _sample_hyperparams(self) -> dict:

        sampled_params = {}
        if self.random_hyperparams == {}:
            return sampled_params
            
        for param, values in self.random_hyperparams.items():

            if isinstance(values, list):

                if param == 'dim_embedding':
                    num_cats = len(self.ts_preprocess.cat_features)
                    sampled_params[param] = [random.choice(values) for _ in range(num_cats)]
                else:
                    sampled_params[param] = random.choice(values)
            
            elif isinstance(values, tuple):
                sampled_params[param] = random.uniform(values[0], values[1])

        return sampled_params
    
    def _set_seed(self, 
                  seed:int
                 ) -> None:
        
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        if self.deterministic:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)

            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        
