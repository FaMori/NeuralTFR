import argparse
import os
import glob
import pandas as pd

from neuraltfr import NeuralTFR
from common.losses import MQLoss
from common.evaluation import eval_models, eval_plot, plot_predictions

def parse_args():
    parser = argparse.ArgumentParser(description="NeuralTFR: Total Fertility Rate (TFR) forecasting using neural networks models.")

    parser.add_argument("--mode", type=str, choices=["eval", "forecast", "all"], default="all",
                        help="Execution mode: 'eval' (evaluation), 'forecast' (future) or 'all' (both).")
    
    parser.add_argument("--data-path", type=str, default="data/final/tfr_smooth.csv",
                        help="Path to the main TFR dataset.")
    parser.add_argument("--out-dir", type=str, default="results/",
                        help="Output directory for plots and metrics.")
    parser.add_argument("--eval-preds-dir", type=str, default="results/evaluation/predictions/other models/",
                        help="Directory containing baseline historical predictions for evaluation (CSV files).")
    parser.add_argument("--forecast-preds-dir", type=str, default="results/forecast/predictions/other models/",
                        help="Directory containing baseline future projections (CSV files).")
    
    parser.add_argument("--enc-len", type=int, default=20,
                        help="Encoder window length.")
    parser.add_argument("--pred-len-eval", type=int, nargs="+", default=[10, 15],
                        help="Prediction length for evaluation mode (list of ints).")
    parser.add_argument("--pred-len-forecast", type=int, nargs="+", default=[10, 15],
                        help="Prediction length for forecast mode (list of ints).")
    parser.add_argument("--n-models", type=int, default=10,
                        help="Number of models in the ensemble.")
    
    parser.add_argument("--eval-split-year", type=int, default=2009,
                        help="Exclusive split year for evaluation training (< eval_split_year).")
    parser.add_argument("--epochs", type=int, default=14,
                        help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.0003,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--n-jobs", type=int, default=5,
                        help="Number of parallel jobs for ensemble training.")
    parser.add_argument("--augment_tfr_level", type=float, default=1.3,
                        help="Duplicate recent windows from series whose historical TFR ever reached a value <= this threshold.")
    parser.add_argument("--augment_n_windows", type=int, default=10,
                        help="Number of most recent windows duplicated per selected series.")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Base random seed for reproducible training.")
    parser.add_argument("--deterministic", type=str2bool, nargs="?", const=True, default=False,
                        help="Enable the most reproducible execution mode possible while keeping parallelization.")
    
    parser.add_argument("--retrain-models", type=str2bool, nargs="?", const=True, default=False,
                        help="Whether to retrain models for forecast mode (if False, use the same evaluation weights for forecast mode).")
    
    return parser.parse_args()

def str2bool(value):
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError("Expected a boolean value (true/false).")

def run_evaluation(args, tfr_data):

    print("\n--- Starting Evaluation ---")

    train_df = tfr_data[tfr_data['year'] < args.eval_split_year].copy()
    train_df['id_serie'] = train_df['id']

    mq_loss = MQLoss(quantiles=[0.05, 0.1, 0.5, 0.9, 0.95])
    
    model = NeuralTFR(
        enc_len=args.enc_len,
        model_name='NeuralTFR',
        pred_len=args.pred_len_eval,
        cat_features=['id_serie'],
        lags_features=[2, 4, 6],
        preprocess_config={'target': 'TFR', 'time_col': 'year', 'id_col': 'id',
                           'apply_log': True, 'scaler': 'Standard'},
        model_config={'enc_hidden_size': 12, 'dec_hidden_size': 12, 'dim_embedding': [6], 
                      'dropout_enc_feat': 0.0, 'dropout_dec_feat': 0.0, 
                      'tf_config': {'tf_type': 'linear', 'tf_max_rate': 0.3, 'tf_decay': 0.035}},
        augment_tfr_level=args.augment_tfr_level,
        augment_n_windows=args.augment_n_windows,
        random_hyperparams={'dim_embedding':[4,6,8], 'enc_hidden_size':[12,16]},
        loss_function=mq_loss,
        n_models=args.n_models,
        seed=args.seed,
        deterministic=args.deterministic
    )

    model.fit(
        train_df, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        stop_patience=8,
        optimizer_config={'lr': args.lr, 'weight_decay': 1.68e-05, 'type_scheduler': 'Exponential', 'gamma': 0.99},
        valid_epochs=0, 
        n_jobs=args.n_jobs
    )

    all_preds = [model.predict(train_df)]
    for file_path in glob.glob(os.path.join(args.eval_preds_dir, "*.csv")):
        preds_df = pd.read_csv(file_path)
        preds_df = preds_df.groupby('id').head(max(args.pred_len_eval)).reset_index(drop=True)
        all_preds.append(preds_df)
    
    preds_df = pd.concat(all_preds, ignore_index=True)
    total_models = preds_df['model'].nunique()
    id_model_counts = preds_df.groupby('id')['model'].nunique()
    eval_ids = id_model_counts[id_model_counts == total_models].index
    eval_ids = eval_ids[eval_ids.isin(tfr_data[tfr_data['year'] >= args.eval_split_year]['id'].unique())]
    preds_df = preds_df[preds_df['id'].isin(eval_ids)].reset_index(drop=True)

    eval_df = eval_models(real_df=tfr_data, 
                          target_col='TFR', 
                          time_col='year', 
                          id_col='id', 
                          reg_col='id_reg',
                          predicts_df=preds_df)
    
    print("\n--- Evaluation Metrics Summary ---")
    metrics_to_summarize = [col for col in eval_df.columns if col not in ['id', 'n']]
    summary_df = eval_df.drop(columns=['id', 'n'], errors='ignore').groupby('model')[metrics_to_summarize].median(numeric_only=True)
    print(summary_df.to_string())
    print("-" * 43)

    os.makedirs(os.path.join(args.out_dir, 'evaluation', 'metrics'), exist_ok=True)
    eval_plot(eval_df, plot_type="raincloud", 
              pdf_path=os.path.join(args.out_dir,'evaluation','metrics', 'metrics.pdf'), show=False)
    eval_df.to_csv(os.path.join(args.out_dir,'evaluation','metrics', 'metrics.csv'), index=False)
    print(f"Evaluation metrics saved to: {os.path.join(args.out_dir, 'evaluation', 'metrics')}")

    os.makedirs(os.path.join(args.out_dir, 'evaluation','predictions'), exist_ok=True)
    plot_predictions(tfr_data, target_col='TFR', id_col='id', time_col='year', 
                     predicts_df=preds_df.drop(columns=['y_hat_10', 'y_hat_90']), 
                     plot_name=os.path.join(args.out_dir, 'evaluation','predictions','plot_evaluation'))
    preds_df.to_csv(os.path.join(args.out_dir, 'evaluation','predictions','predictions.csv'), index=False)
    print(f"Prediction plots saved to: {os.path.join(args.out_dir, 'evaluation', 'predictions')}.pdf")
    
    return model

def run_forecast(args, tfr_data, shared_model=None):

    print("\n--- Starting Forecasting ---")

    tfr_data['id_serie'] = tfr_data['id']
    forecast_min_pred_len = min(args.pred_len_forecast)
    forecast_max_pred_len = max(args.pred_len_forecast)
    reuse_shared_model = (
        shared_model is not None
        and not args.retrain_models
        and shared_model.min_pred_len == forecast_min_pred_len
        and shared_model.max_pred_len == forecast_max_pred_len
    )

    if reuse_shared_model:
        print("Reusing evaluation weights for forecast mode (--retrain-models=False).")
        model = shared_model
    else:
        if shared_model is None and not args.retrain_models:
            print("No evaluation model available to reuse; training forecast models.")
        elif not args.retrain_models:
            print("Forecast horizon differs from evaluation horizon; retraining forecast models.")

        mq_loss = MQLoss(quantiles=[0.05, 0.1, 0.5, 0.9, 0.95])
        
        model = NeuralTFR(
            enc_len=args.enc_len,
            model_name='NeuralTFR',
            pred_len=args.pred_len_forecast,
            cat_features=['id_serie'],
            lags_features=[2, 4, 6],
            preprocess_config={'target': 'TFR', 'time_col': 'year', 'id_col': 'id',
                               'apply_log': True, 'scaler': 'Standard'},
            model_config={'enc_hidden_size': 12, 'dec_hidden_size': 12, 'dim_embedding': [6], 
                          'dropout_enc_feat': 0.0, 'dropout_dec_feat': 0.0, 
                          'tf_config': {'tf_type': 'linear', 'tf_max_rate': 0.3, 'tf_decay': 0.035}},
            augment_tfr_level=args.augment_tfr_level,
            augment_n_windows=args.augment_n_windows,
            random_hyperparams={'dim_embedding': [4, 6, 8], 'enc_hidden_size': [8, 12, 16]},
            loss_function=mq_loss,
            n_models=args.n_models,
            seed=args.seed,
            deterministic=args.deterministic
        )

        model.fit(
            tfr_data, 
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            stop_patience=8,
            optimizer_config={'lr': args.lr, 'weight_decay': 1.68e-05, 'type_scheduler': 'Exponential', 'gamma': 0.99},
            valid_epochs=0, 
            n_jobs=args.n_jobs
        )

    preds = model.predict(tfr_data)

    os.makedirs(os.path.join(args.out_dir, 'forecast','predictions'), exist_ok=True)
    plot_predictions(tfr_data, target_col='TFR', id_col='id', time_col='year', 
                     predicts_df=preds.drop(columns=['y_hat_10','y_hat_90']), 
                     plot_name=os.path.join(args.out_dir, 'forecast','predictions','plot_forecast'))
    preds.to_csv(os.path.join(args.out_dir, 'forecast','predictions','predictions.csv'), index=False)
    print(f"Forecast prediction plots saved to: {os.path.join(args.out_dir, 'forecast','predictions','plot_forecast')}.pdf")

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'evaluation'), exist_ok=True)
    print("="*50)
    print(f"NeuralTFR - Mode: {args.mode.upper()}")
    print("="*50)
    
    tfr_data = pd.read_csv(args.data_path)
    
    eval_model = None
    if args.mode in ["eval", "all"]:
        eval_model = run_evaluation(args, tfr_data)
        
    if args.mode in ["forecast", "all"]:
        run_forecast(args, tfr_data, shared_model=eval_model)

if __name__ == "__main__":
    main()
