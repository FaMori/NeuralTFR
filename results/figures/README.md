# Figures

The numbers behind the manuscript's figures, appendix analyses and the text
passages that cite numbers no figure shows, written by `python figures.py`
after `evaluation.py`, and the figures as published in the manuscript (PDF).
The repository
does not draw the figures: each table holds exactly what its figure shows, so
that the figure can be redrawn or read without the drawing code.

Unless a table says otherwise, populations are the cohort of `evaluation`:
the 144 with complete outcomes at every origin in every model. Fertility
groups at an origin use the reported annual median of that year: very low
below 1.5, moderately low from 1.5 to 2.1, at or above replacement from 2.1.

| figure | tables | what they hold |
|---|---|---|
| 2 (paired differences), Tables 1 and E.2 | `../paired_comparisons.csv` | the paired comparisons of `evaluation.py` |
| 4 (fertility change) | `fig4_change_by_range.csv`, `fig4_zero_crossings.csv` | mean fifteen-year change, observed and forecast by each model, in ranges of starting fertility placed at their mean starting level, with a population-bootstrap band for the observed change; the level at which each series first turns negative, with its bootstrap interval |
| 5, G1, G2 (trajectories) | `fig5_trajectories_forecasts.csv`, `fig5_trajectories_observed.csv` | the five models' forecasts from 2006 and the reported values 1995-2021 for the seventeen illustrated populations; `display` says whether a population is in the main figure or the appendix |
| 6 (forward projections) | `fig6_forward_by_year.csv`, `fig6_forward_by_group.csv`, `fig6_forward_2038.csv` | for the 130 populations with a 2024-2038 projection from NeuralTFR, the regressions, UN WPP 2024 and GBD 2021: mean and population-weighted mean TFR by year, counts below 1.3, 1.5 and 2.1 (and expected counts from the quantiles where available); change to 2038 and counts by fertility group in 2023; every population's 2038 values |
| B1 (composition) | `figB1_composition.csv` | populations with a reported value each year, by fertility group |
| E1 (interval groups), Appendix D | `figE1_interval_groups.csv`, `appendixD_miss_direction.csv` | coverage, width and Winkler score of the 80% and 95% intervals by model, fertility group at the origin and horizon band (means over five years per population-origin, then over pairs); the shares of observations below and above the intervals and the mean error of the median, overall and by group |
| F1 (historical performance) | `figF1_populations.csv`, `figF1_medians.csv` | each population's scores averaged over origins, and their medians pooled and by origin |
| Appendix C (what the network responds to) | `appendixC_grid_responses.csv`, `appendixC_flat_history*.csv` | the median forecast, across the ten fits at each origin, for simulated histories ending at a level after a constant annual change; the loss of the fits when each observed history is replaced by its final value, overall, by origin and group with bootstrap intervals, by fit, and by horizon |
| Table E.3, Section 3.1 and Appendix D comparisons within fertility groups | `../group_comparisons.csv` | the comparisons within fertility groups of `evaluation.py` |
| text: very low fertility (Sections 3.1-3.2, Appendix D, Discussion) | `text_very_low_pairs.csv`, `text_very_low_counts.csv` | each of the 90 population-origin pairs that started below 1.5: fertility at the origin, at year 15 and in the latest data, the year of the highest value after the origin, and the mean sMAPE, error of the median (all years and years 11-15) and 95% Winkler score of NeuralTFR, the spline and BayesTFR; the counts and means the text cites from it |
| text: Introduction | `text_recovery_panel.csv`, `text_very_low_counts.csv` | every population of the panel below 1.5 by 2006: its lowest level by then, its highest level after that low, its latest value and its 2023 population; the counts the text cites from it |
