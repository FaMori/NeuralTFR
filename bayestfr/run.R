#!/usr/bin/env Rscript
# One re-estimation of BayesTFR at a forecast origin, as recorded for the paper.
#
#   Rscript run.R <origin> [phase3_extension]       # from this directory

library(bayesTFR)

args <- commandArgs(trailingOnly = TRUE)
origin <- as.integer(args[[1]])
extension <- if (length(args) > 1) as.integer(args[[2]]) else 0L

HORIZON <- 15
CHAINS <- 3
SEED <- 1234
ITER2 <- 65000; THIN2 <- 1;  BURNIN2 <- 5000
ITER3 <- 55000; THIN3 <- 10; BURNIN3 <- 5000
TRAJECTORIES <- 2000

input <- sprintf("input/empirical_%d.txt", origin)
sims <- sprintf("runs/%d/sims", origin)
preds <- sprintf("runs/%d/preds", origin)
results <- "../results/bayestfr"
dir.create(sims, recursive = TRUE, showWarnings = FALSE)
dir.create(results, recursive = TRUE, showWarnings = FALSE)

run.tfr.mcmc(nr.chains = CHAINS, iter = ITER2, thin = THIN2, output.dir = sims,
             annual = TRUE, uncertainty = FALSE, ar.phase2 = TRUE,
             my.tfr.file = input, start.year = 1950, present.year = origin,
             seed = SEED, parallel = TRUE, nr.nodes = CHAINS, verbose = FALSE)
run.tfr3.mcmc(sim.dir = sims, nr.chains = CHAINS, iter = ITER3, thin = THIN3,
              seed = SEED, parallel = TRUE, nr.nodes = CHAINS, verbose = FALSE)
if (extension > 0)
  continue.tfr3.mcmc(iter = extension, sim.dir = sims, parallel = TRUE,
                     nr.nodes = CHAINS, verbose = FALSE)

diagnostics <- c(
  "--- Phase II ---",
  capture.output(print(summary(tfr.diagnose(sims, thin = THIN2, burnin = BURNIN2, express = FALSE)))),
  "",
  "--- Phase III ---",
  capture.output(print(summary(tfr3.diagnose(sims, thin = THIN3, burnin = BURNIN3, express = FALSE)))))
writeLines(diagnostics, file.path(results, sprintf("diagnostics_%d.txt", origin)))

tfr.predict(sim.dir = sims, end.year = origin + HORIZON, burnin = BURNIN2,
            nr.traj = TRAJECTORIES, use.tfr3 = TRUE, burnin3 = BURNIN3,
            output.dir = preds, seed = SEED, verbose = FALSE)
file.copy(file.path(preds, "predictions", "projection_summary_user_friendly.csv"),
          file.path(results, sprintf("summary_%d.csv", origin)), overwrite = TRUE)
