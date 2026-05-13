import pandas as pd
import numpy as np
import re
import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker
from matplotlib.ticker import PercentFormatter, AutoMinorLocator
import seaborn as sns
from scipy import stats

def plot_predictions(real_df:pd.DataFrame,
                     target_col:str,
                     id_col:str,
                     time_col:str,
                     predicts_df:pd.DataFrame,
                     plot_name:str='plots_predictions',
                     fig_size:tuple=(12,7),
                     first_ds:int=1975
                     ) -> None:

    with PdfPages(f'{plot_name}.pdf') as pdf_pages:

        ids_to_plot = sorted(predicts_df[id_col].unique())
        models_to_plot = sorted(predicts_df['model'].unique())

        colors = plt.cm.viridis(np.linspace(0, 1, len(models_to_plot)))
        model_color_map = {model: color for model, color in zip(models_to_plot, colors)}

        for serie_id in ids_to_plot:

            real_serie = real_df[(real_df[id_col] == serie_id) & (real_df[time_col] >= first_ds)]
            pred_serie = predicts_df[predicts_df[id_col] == serie_id]

            serie_name = real_serie['name'].values[0] if 'name' in real_serie.columns else serie_id

            fig, ax = plt.subplots(figsize=fig_size)
        
            ax.scatter(real_serie[time_col], real_serie[target_col], label='Real Data', color='black', s=15, alpha=0.9, zorder=4)

            for model in models_to_plot:

                model_pred_serie = pred_serie[pred_serie['model'] == model].sort_values(time_col)
                color = model_color_map[model]
                
                quantile_map = {}
                for col in model_pred_serie.columns:
                    if 'y_hat_' in col:
                        quantile_val = float(re.findall(r'[\d\.]+', col)[0])
                        if quantile_val > 1: 
                            quantile_val /= 100
                        quantile_map[quantile_val] = col

                lower_quantiles = sorted([q for q in quantile_map.keys() if q < 0.5])
                
                for i, lq in enumerate(lower_quantiles):
                    uq = 1.0 - lq
                    if uq in quantile_map:
                        lower_col = quantile_map[lq]
                        upper_col = quantile_map[uq]
                        interval_pct = (uq - lq) * 100
                        
                        ax.fill_between(
                            model_pred_serie[time_col],
                            model_pred_serie[lower_col],
                            model_pred_serie[upper_col],
                            color=color,
                            alpha=0.15 + (i * 0.1),
                            label=f'{model} - {interval_pct:.0f}% Prediction Interval',
                            zorder=2
                        )

                median_col = next((col for col in ['y_hat_50'] if col in model_pred_serie.columns), None)
                if median_col and not model_pred_serie[median_col].isnull().all():
                    ax.plot(
                        model_pred_serie[time_col],
                        model_pred_serie[median_col],
                        linestyle='--',
                        color=color,
                        label=f'{model} - Median Forecast',
                        zorder=3
                    )

            title_text = f'Forecast for {serie_name}'
            subtitle_text = f'Prediction Horizon: {pred_serie[time_col].min()} to {pred_serie[time_col].max()}'
            ax.set_title(f'{title_text}\n{subtitle_text}', fontsize=14)
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel(target_col, fontsize=12)
            ax.legend(loc='best')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            fig.tight_layout()

            pdf_pages.savefig(fig, bbox_inches='tight')
            plt.close(fig)

def eval_models(real_df:pd.DataFrame,
                target_col:str,
                id_col:str,
                time_col:str,
                reg_col:str,
                predicts_df:pd.DataFrame
               ) -> tuple[pd.DataFrame, pd.DataFrame|None]:
    
    eval_df = (
        real_df[[id_col, time_col, reg_col, target_col]]
        .copy()
        .sort_values([id_col, time_col])
        .merge(predicts_df, on=[id_col, time_col], how='left')
    )
            
    pred_cols = [c for c in predicts_df.columns if c.startswith('y_hat_')]
    perc_map = {}
    for c in pred_cols:
        m = re.match(r'y_hat_(\d{1,3})$', c)
        if m:
            p = int(m.group(1))
            perc_map[p] = c

    coverage_pairs = []
    for p in sorted([x for x in perc_map.keys() if 0 < x < 50]):
        q = 100 - p
        if q in perc_map:
            coverage_pairs.append((p, q, perc_map[p], perc_map[q]))

    def _rmse(y,y_hat):
        return np.sqrt(np.mean((y_hat - y)**2))
    
    def _smape(y,y_hat):
        return 100/len(y) * np.sum(2 * np.abs(y_hat - y) / (np.abs(y) + np.abs(y_hat) + 1e-8))
        
    def _rmsse(y,y_hat,y_train):
        num = np.sum((y - y_hat)**2)
        den = np.mean(np.square(np.diff(y_train))) + 1e-8
        rmsse = np.sqrt(num / (len(y) * den))
        return rmsse
            
    def _pinball_loss(y, y_hat, alpha):
        return np.maximum(alpha * (y - y_hat), (alpha - 1) * (y - y_hat))
    
    def _crps_approx(y, quantiles_dict):
        alphas = sorted(quantiles_dict.keys())
        if len(alphas) < 2:
            if not alphas: return np.nan
            return np.mean(_pinball_loss(y, quantiles_dict[alphas[0]], alphas[0])) * 2
        crps_total = np.zeros(len(y))
        for i in range(len(alphas) - 1):
            a1, a2 = alphas[i], alphas[i+1]
            loss1 = _pinball_loss(y, quantiles_dict[a1], a1)
            loss2 = _pinball_loss(y, quantiles_dict[a2], a2)
            crps_total += (a2 - a1) * (loss1 + loss2) / 2.0
        return np.mean(crps_total) * 2
    
    def _picp(y, y_lo, y_hi):
        return np.mean((y >= y_lo) & (y <= y_hi)) * 100

    def _mis(y, y_lo, y_hi, alpha):
        width = y_hi - y_lo
        penalty_lo = (2.0 / alpha) * (y_lo - y) * (y < y_lo)
        penalty_hi = (2.0 / alpha) * (y - y_hi) * (y > y_hi)
        return np.mean(width + penalty_lo + penalty_hi)
    
    rows = []
    for serie_id in predicts_df[id_col].unique():
        serie_eval = eval_df[eval_df[id_col] == serie_id]

        y_train = serie_eval[serie_eval['y_hat_50'].isna()][target_col].values

        for model in predicts_df['model'].unique():
            model_eval = serie_eval[serie_eval['model'] == model]
            
            y = model_eval[target_col].values
            y50 = model_eval['y_hat_50'].values
            quantiles = {p/100: model_eval[c].values for p, c in perc_map.items()}

            row = {
                'id': serie_id,
                'model': model,
                'n': int(len(model_eval)),
                'rmse': _rmse(y, y50),
                'smape': _smape(y, y50),
                'rmsse': _rmsse(y, y50, y_train),
                'crps': _crps_approx(y, quantiles)
            }

            for p, q, c_lo, c_hi in coverage_pairs:
                width_pct = q - p  
                alpha = 1.0 - (width_pct / 100.0)
                
                y_lo = model_eval[c_lo].values
                y_hi = model_eval[c_hi].values
                
                row[f'coverage_{width_pct}'] = _picp(y, y_lo, y_hi)
                row[f'mpiw_{width_pct}'] = np.mean(y_hi - y_lo)
                row[f'mis_{width_pct}'] = _mis(y, y_lo, y_hi, alpha)
    
            rows.append(row)

    metrics_df = pd.DataFrame(rows)

    return metrics_df

def eval_plot(df: pd.DataFrame,
              plot_type: str = "boxplot", 
              pdf_path: str | None = None,
              show: bool = True,
              model_col: str = "model",
              id_col: str = "id",
              palette_name: str = "warm_red") -> dict:
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 10
    })
    sns.set_theme(style='whitegrid', context='paper')

    point_metrics = [m for m in ['rmse', 'smape', 'rmsse'] if m in df.columns]
    uncertainty_metrics = [m for m in ['crps', 'coverage_90', 'mpiw_90'] if m in df.columns]

    palette_options = {
        'warm_red': {
            'NeuralTFR': '#B33A3A',
            'BayesTFR': '#3FA34D',
            'NaiveDrift': '#9A9A9A'
        },
        'deep_contrast': {
            'NeuralTFR': '#9F2F2F',
            'BayesTFR': '#2E8B57',
            'NaiveDrift': '#8F8F8F'
        },
        'soft_editorial': {
            'NeuralTFR': '#FF3400',
            'BayesTFR': '#99FF00',
            'NaiveDrift': '#82ECFF'
        }
    }
    palette = palette_options.get(palette_name, palette_options['warm_red'])
    contour_palette = {
        'NeuralTFR': '#FC5000',
        'BayesTFR': '#00C745',
        'NaiveDrift': '#00ACF5'
    }

    model_order = [m for m in ['NeuralTFR', 'BayesTFR', 'NaiveDrift'] if m in df[model_col].dropna().unique()]
    model_order += [m for m in df[model_col].dropna().unique() if m not in model_order]

    def _format_title(metric: str) -> str:
        m = metric.lower()
        title_map = {
            'rmse': 'RMSE',
            'smape': 'sMAPE (%)',
            'rmsse': 'RMSSE',
            'crps': 'CRPS',
            'coverage_90': 'Coverage 90% (%)',
            'coverage_80': 'Coverage 80% (%)',
            'mpiw_90': 'MPIW 90%',
            'mpiw_80': 'MPIW 80%',
            'mis_90': 'MIS 90%',
            'mis_80': 'MIS 80%'
        }
        return title_map.get(m, metric.upper())

    def _value_formatter(metric: str) -> str:
        return '{:.1f}' if metric.startswith('coverage_') or metric == 'smape' else '{:.3f}'

    def _get_significance_text(data: pd.DataFrame, metric: str) -> str:
        if id_col not in data.columns:
            return ""

        clean_data = data.dropna(subset=[metric, id_col, model_col])
        if clean_data.empty:
            return ""

        try:
            pivot_df = clean_data.pivot(index=id_col, columns=model_col, values=metric).dropna()
        except ValueError:
            pivot_df = clean_data.pivot_table(index=id_col, columns=model_col, values=metric, aggfunc='mean').dropna()

        if pivot_df.shape[1] < 2:
            return ""

        try:
            medians = pivot_df.median().sort_values()
            best_model = medians.index[0]
            second_best = medians.index[1]

            if pivot_df.shape[1] >= 3:
                _, p_friedman = stats.friedmanchisquare(*[pivot_df[m] for m in pivot_df.columns])
                if p_friedman >= 0.05:
                    return f'Best median: {best_model}\nFriedman p={p_friedman:.3f}'

            _, p_wilcoxon = stats.wilcoxon(pivot_df[best_model], pivot_df[second_best], alternative='less')
            return f'Best median: {best_model}\nWilcoxon p={p_wilcoxon:.3f}'
        except Exception:
            return ""

    def _style_axis(ax, metric: str | None = None):
        ax.grid(axis='x', which='major', color='#D7DEE7', linewidth=0.8, alpha=0.75)
        ax.grid(axis='x', which='minor', color='#E6EBF2', linewidth=0.6, alpha=0.95)
        ax.grid(axis='y', visible=False)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        for spine_name in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine_name].set_visible(True)
            ax.spines[spine_name].set_color('#C9D2DE')
            ax.spines[spine_name].set_linewidth(0.55)
        ax.spines['bottom'].set_color('#9AA4B2')
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['bottom'].set_zorder(0)
        ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', which='minor', length=0)
        ax.tick_params(axis='x', labelsize=10.0, colors='#475569')
        ax.xaxis.set_ticks_position('bottom')
        ax.tick_params(axis='x', top=False, labeltop=False, bottom=True, labelbottom=True, pad=2)
        ax.set_facecolor('#FBFCFE')

    def _darken_color(color: str, amount: float = 0.8) -> tuple:
        rgb = np.array(mcolors.to_rgb(color))
        return tuple(np.clip(rgb * amount, 0, 1))

    def _draw_distribution(ax, data: pd.DataFrame, metric: str, panel_label: str | None = None):
        valid_data = data.dropna(subset=[metric]).copy()
        if metric in uncertainty_metrics:
            valid_data = valid_data[valid_data[model_col] != 'NaiveDrift'].copy()
        if valid_data.empty:
            ax.set_visible(False)
            return

        metric_model_order = [m for m in model_order if m in valid_data[model_col].unique()]

        sns.violinplot(
            data=valid_data,
            x=metric,
            y=model_col,
            hue=model_col,
            order=metric_model_order,
            palette=palette,
            legend=False,
            dodge=False,
            inner=None,
            cut=0,
            linewidth=0.12,
            saturation=1,
            ax=ax
        )
        violin_collections = list(ax.collections)
        for idx, collection in enumerate(violin_collections[:len(metric_model_order)]):
            model_name = metric_model_order[idx]
            base_color = palette.get(model_name, '#374151')
            edge_color = contour_palette.get(model_name, '#374151')
            collection.set_facecolor(mcolors.to_rgba(base_color, 0.14))
            collection.set_edgecolor(mcolors.to_rgba(edge_color, 0.88))
            collection.set_linewidth(0.65)
            collection.set_alpha(None)

        sns.stripplot(
            data=valid_data,
            x=metric,
            y=model_col,
            hue=model_col,
            palette=palette,
            order=metric_model_order,
            dodge=False,
            legend=False,
            alpha=0.28,
            size=3.2,
            jitter=0.16,
            ax=ax
        )
        scatter_collections = ax.collections[len(metric_model_order):]
        for idx, collection in enumerate(scatter_collections[:len(metric_model_order)]):
            model_name = metric_model_order[idx]
            base_color = palette.get(model_name, '#374151')
            edge_color = contour_palette.get(model_name, '#374151')
            collection.set_facecolor(mcolors.to_rgba(base_color, 0.30))
            collection.set_edgecolor(mcolors.to_rgba(edge_color, 0.30))
            collection.set_linewidth(0.20)
            collection.set_alpha(None)

        grouped = valid_data.groupby(model_col)[metric]
        summary = pd.DataFrame({
            'p10': grouped.quantile(0.10),
            'q1': grouped.quantile(0.25),
            'median': grouped.median(),
            'q3': grouped.quantile(0.75),
            'p90': grouped.quantile(0.90)
        }).reindex(metric_model_order).dropna()

        for idx, (model_name, row) in enumerate(summary.iterrows()):
            ax.hlines(
                y=idx,
                xmin=row['p10'],
                xmax=row['p90'],
                color=mcolors.to_rgba(contour_palette.get(model_name, '#374151'), 0.34),
                linewidth=0.52,
                zorder=4
            )
            ax.vlines(
                x=[row['p10'], row['p90']],
                ymin=idx - 0.035,
                ymax=idx + 0.035,
                color=mcolors.to_rgba(contour_palette.get(model_name, '#374151'), 0.34),
                linewidth=0.52,
                zorder=4
            )
            ax.hlines(
                y=idx,
                xmin=row['q1'],
                xmax=row['q3'],
                color=mcolors.to_rgba(contour_palette.get(model_name, '#374151'), 0.72),
                linewidth=0.95,
                zorder=5
            )
            ax.vlines(
                x=[row['q1'], row['q3']],
                ymin=idx - 0.05,
                ymax=idx + 0.05,
                color=mcolors.to_rgba(contour_palette.get(model_name, '#374151'), 0.72),
                linewidth=0.95,
                zorder=5
            )
            ax.scatter(
                row['median'],
                idx,
                s=56,
                color=mcolors.to_rgba(palette.get(model_name, '#374151'), 0.95),
                edgecolor=mcolors.to_rgba(contour_palette.get(model_name, '#374151'), 0.92),
                linewidth=0.55,
                zorder=6
            )

        if metric.startswith('coverage_'):
            nominal = float(metric.split('_')[1])
            ax.axvline(nominal, color='#C2410C', linestyle=(0, (4, 3)), linewidth=1.1, alpha=0.9)
            ax.text(
                nominal,
                0.985,
                f'Nominal {int(nominal)}%',
                transform=ax.get_xaxis_transform(),
                ha='center',
                va='top',
                fontsize=10.0,
                color='#9A3412',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.84, pad=0.2)
            )
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
            ax.set_xticks(np.arange(0, 101, 20))

        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_yticklabels([])
        _style_axis(ax, metric)

        best_model_name = None
        significance_stars = ''
        sig_text = _get_significance_text(valid_data, metric)
        if sig_text:
            best_match = re.search(r'Best median: ([^\n]+)', sig_text)
            if best_match:
                best_model_name = best_match.group(1).strip()
            p_match = re.search(r'p=(\d*\.?\d+)', sig_text)
            if p_match:
                p_value = float(p_match.group(1))
                if p_value < 0.001:
                    significance_stars = '***'
                elif p_value < 0.01:
                    significance_stars = '**'
                elif p_value < 0.05:
                    significance_stars = '*'

        legend_rows = []
        for model_name in metric_model_order:
            label = model_name
            swatch = DrawingArea(26, 17, 0, 0)
            swatch.add_artist(
                Rectangle(
                    (1.0, 2.6),
                    16.6,
                    9.9,
                    facecolor=mcolors.to_rgba(palette.get(model_name, '#374151'), 0.14),
                    edgecolor=mcolors.to_rgba(contour_palette.get(model_name, '#374151'), 0.88),
                    linewidth=0.75
                )
            )
            legend_rows.append(
                HPacker(
                    children=[
                        swatch,
                        TextArea(label, textprops={'fontsize': 16.0, 'color': '#374151'})
                    ],
                    align='baseline',
                    pad=0,
                    sep=6.0
                )
            )
        if sig_text:
            test_note = sig_text.split('\n')[-1].strip()
            note_label = f'{significance_stars} {test_note}'.strip()
            legend_rows.append(
                TextArea(note_label, textprops={'fontsize': 13.8, 'color': '#374151'})
            )
            legend_box = VPacker(children=legend_rows, align='left', pad=0, sep=4.0)
        else:
            legend_box = VPacker(children=legend_rows, align='left', pad=0, sep=4.0)
        legend_loc = 'upper left' if metric == 'coverage_90' else 'upper right'
        legend_anchor = (0.03, 0.99) if metric == 'coverage_90' else (0.985, 0.99)
        anchored_box = AnchoredOffsetbox(
            loc=legend_loc,
            child=legend_box,
            pad=0.22,
            frameon=True,
            bbox_to_anchor=legend_anchor,
            bbox_transform=ax.transAxes,
            borderpad=0.4
        )
        anchored_box.patch.set_facecolor('white')
        anchored_box.patch.set_edgecolor('#D6DCE5')
        anchored_box.patch.set_alpha(0.96)
        ax.add_artist(anchored_box)

        x_min = valid_data[metric].min()
        x_max = valid_data[metric].max()
        x_range = max(x_max - x_min, 1e-8)
        if metric.startswith('coverage_'):
            ax.set_xlim(-6, 106)
        else:
            ax.set_xlim(x_min - 0.07 * x_range, x_max + 0.16 * x_range)
        ax.set_ylim(len(metric_model_order) - 0.30, -1.02)

        if panel_label:
            ax.text(-0.1, 1.08, panel_label, transform=ax.transAxes,
                    fontsize=13, fontweight='bold', ha='left', va='bottom', color='#111827')

    def _save_figure(fig, path_without_ext: str, saved_paths: list[str]):
        pdf_out = f'{path_without_ext}.pdf'
        png_out = f'{path_without_ext}.png'
        fig.savefig(pdf_out, dpi=300)
        fig.savefig(png_out, dpi=300)
        saved_paths.extend([pdf_out, png_out])

    saved_paths = []
    all_metrics = point_metrics + uncertainty_metrics

    for metric in all_metrics:
        fig_size = (7.8, 2.9)
        fig, ax = plt.subplots(figsize=fig_size)
        fig.subplots_adjust(left=0.02, right=0.998, top=0.99, bottom=0.07)
        _draw_distribution(ax, df, metric)

        if pdf_path:
            base_dir, file_name = os.path.split(pdf_path)
            name, _ = os.path.splitext(file_name)
            _save_figure(fig, os.path.join(base_dir, f'{name}_{metric}'), saved_paths)

        if show:
            plt.show()
        else:
            plt.close(fig)

    return {
        'point_metrics': point_metrics,
        'uncertainty_metrics': uncertainty_metrics,
        'plot_type': 'paper_standalone',
        'palette_name': palette_name,
        'saved_to': saved_paths
    }
