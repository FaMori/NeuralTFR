# BayesTFR

The Bayesian hierarchical model of fertility change behind the United Nations
projections (Alkema et al. 2011; R package `bayesTFR`), re-estimated at each
historical origin on the same panel the other models are fitted to, with
nothing read from after the origin. Its median and its 80% and 95% prediction
intervals, taken from 2,000 simulated trajectories, are the benchmark the
manuscript compares NeuralTFR with.

```
input/empirical_<origin>.txt   the panel cut at the origin.
input/countries_short.csv      populations with fewer than 25 years at an origin, left out there
run.R                          one re-estimation: MCMC, diagnostics, prediction
tidy.py                        the recorded summaries in the layout of the other models' forecasts
```

## Configuration

| | |
|---|---|
| chains | 3, run in parallel within one process |
| phase II (`run.tfr.mcmc`) | 65,000 iterations, thin 1, burn-in 5,000; annual data from 1950, `ar.phase2 = TRUE`, no data uncertainty |
| phase III (`run.tfr3.mcmc`) | 55,000 iterations, thin 10, burn-in 5,000 |
| prediction (`tfr.predict`) | 2,000 trajectories, phase III used, 15 years from the origin |
| seed | 1234 |
