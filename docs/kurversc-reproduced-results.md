# KurveRSC: maintainer-reproduced results

RelArena reproduced all 21 RelBench v1 tasks at seed 0 with
`kurversc[staged]==0.2.0` and `graphreduce==1.10.18`. The system is
`kurversc-2026-10-22` (`kurversc-latest`).

Snapshot: [`fcab8fb`](https://github.com/PriorLabs/relarena/blob/fcab8fb46b82299321c101b9a921d9a3f7c72263), November 5, 2026.
[Source scores](https://github.com/PriorLabs/relarena/blob/fcab8fb46b82299321c101b9a921d9a3f7c72263/baseline_results/results.csv) ·
[Source leaderboard](https://github.com/PriorLabs/relarena/blob/fcab8fb46b82299321c101b9a921d9a3f7c72263/README.md#-leaderboards) ·
[Full-precision extract](assets/kurversc-relarena-reproduction-20261105.json).

## Combined leaderboard

KurveRSC is **#2 of 16 overall and the highest-ranked system**: **1913.4 Elo**
(95% task-bootstrap interval **1832.3-2050.6**). The latest TabPFN-Rel client is
first at **1933.6** (**1864.7-2033.5**). These are point rankings with overlapping
intervals, based on 100 task-bootstrap samples and a constant-global anchor of
1000. They are not the separately computed models-only ratings.

| Rank | Method | Kind | Elo | 95% task-bootstrap interval |
|---:|:--------------------------------|:------|------:|:---------------------------|
| 1 | tabpfn-rel-client-latest | Model | 1933.6 | 1864.7-2033.5 |
| 2 | kurversc-latest | System | 1913.4 | 1832.3-2050.6 |
| 3 | tabpfn-rel-local-latest | Model | 1887.7 | 1831.3-1974.8 |
| 4 | rt-j | System | 1842.2 | 1721.9-1997.8 |
| 5 | tabpfn-rel-client-2026-08-15 | Model | 1799.0 | 1740.7-1886.9 |
| 6 | rt-plurel | System | 1787.7 | 1695.9-1889.4 |
| 7 | kurversc-2026-09-03 | System | 1773.8 | 1709.9-1849.6 |
| 8 | nori-rel | Model | 1732.3 | 1660.5-1844.7 |
| 9 | tabpfn-rel-local-2026-08-15 | Model | 1702.0 | 1613.7-1771.7 |
| 10 | graphsage | Model | 1660.3 | 1550.8-1747.2 |
| 11 | relgt | Model | 1562.7 | 1430.3-1664.5 |
| 12 | rdblearn | Model | 1529.8 | 1437.8-1610.5 |
| 13 | relgnn-es | Model | 1525.0 | 1421.2-1636.6 |
| 14 | lightgbm | Model | 1363.9 | 1168.9-1501.1 |
| 15 | constant-per-entity | Model | 1266.5 | 1082.8-1408.5 |
| 16 | constant-global | Model | 1000.0 | 838.5-1076.9 |

## Classification: ROC-AUC, higher is better

Earlier KurveRSC is `kurversc-2026-09-03`; the client is
`tabpfn-rel-client-2026-09-28`. All rows below use selected held-out test scores.
Winners use full precision before display rounding.

| Dataset / task | Earlier KurveRSC | KurveRSC 0.2.0 | TabPFN-Rel client | Winner |
|:--------------------------|-------------:|-------------:|-------------:|:---------|
| Amazon / item-churn | 0.827256 | 0.828712 | 0.828653 | KurveRSC |
| Amazon / user-churn | 0.710347 | 0.712558 | 0.711698 | KurveRSC |
| Avito / user-clicks | 0.658406 | 0.660062 | 0.691631 | Client |
| Avito / user-visits | 0.674721 | 0.672515 | 0.672453 | KurveRSC |
| Event / user-ignore | 0.832628 | 0.850888 | 0.784037 | KurveRSC |
| Event / user-repeat | 0.779679 | 0.782152 | 0.760495 | KurveRSC |
| F1 / driver-dnf | 0.746484 | 0.722017 | 0.727809 | Client |
| F1 / driver-top3 | 0.664957 | 0.834154 | 0.791702 | KurveRSC |
| H&M / user-churn | 0.696916 | 0.707174 | 0.706021 | KurveRSC |
| Stack / user-badge | 0.877626 | 0.876020 | 0.883443 | Client |
| Stack / user-engagement | 0.903027 | 0.903861 | 0.907662 | Client |
| Trial / study-outcome | 0.712131 | 0.760476 | 0.764556 | Client |

## Regression: MAE, lower is better

| Dataset / task | Earlier KurveRSC | KurveRSC 0.2.0 | TabPFN-Rel client | Winner |
|:--------------------------|-------------:|-------------:|-------------:|:---------|
| Amazon / item-ltv | 42.2731 | 42.3828 | 45.2705 | KurveRSC |
| Amazon / user-ltv | 14.0205 | 13.9894 | 13.8771 | Client |
| Avito / ad-ctr | 0.033440 | 0.033258 | 0.031258 | Client |
| Event / user-attendance | 0.258668 | 0.234934 | 0.241652 | KurveRSC |
| F1 / driver-position | 3.8632 | 4.0605 | 3.7574 | Client |
| H&M / item-sales | 0.032168 | 0.033636 | 0.036479 | KurveRSC |
| Stack / post-votes | 0.063243 | 0.062864 | 0.062733 | Client |
| Trial / site-success | 0.403842 | 0.398749 | 0.377615 | Client |
| Trial / study-adverse | 41.2702 | 33.1750 | 34.4014 | KurveRSC |

There are **11 wins and 10 losses** against the latest client. Relative to the
earlier KurveRSC registration, 15 scores improve and 6 regress. These are
observed results at one seed, not a per-component ablation or a guarantee of
future improvement. See [measured resources](kurversc-staged-resources.md) and the
[technical report](kurversc-staged-technical-report.md).

## Sources for the compared methods

These identify the underlying methods, not alternate sources for the reproduced
scores above. The registered variants and fitting policies are defined by the
frozen RelArena snapshot.

| Method / family | Primary papers or official releases |
|:----------------|:------------------------------------|
| RT-PluRel | [Relational Transformer](https://arxiv.org/abs/2510.06377); [PluRel](https://proceedings.mlr.press/v306/kothapalli26a.html). |
| RT-J | [RT-J](https://arxiv.org/abs/2610.17867), building on the RT and PluRel papers above. |
| TabPFN-Rel API/local | [RelArena / TabPFN-Rel](https://arxiv.org/abs/2608.16319v2); [TabPFN-3](https://arxiv.org/abs/2605.13986v2); [TabPFN-3.5](https://arxiv.org/abs/2609.17895), according to the dated submission. |
| Nori-Rel | [Nori software citation](https://doi.org/10.5281/zenodo.20710462); [official Nori-Rel release](https://www.synthefy.com/blog/nori-rel). |
| GraphSAGE | [GraphSAGE](https://arxiv.org/abs/1706.02216); [RelBench implementation](https://arxiv.org/abs/2407.20060); [relational deep learning](https://proceedings.mlr.press/v235/fey24a.html). |
| RelGT | [Relational Graph Transformer](https://arxiv.org/abs/2505.10960v2). |
| RDBLearn | [RDBLearn](https://arxiv.org/abs/2602.18495). |
| RelGNN | [RelGNN](https://arxiv.org/abs/2502.06784); the leaderboard uses the `relgnn-es` variant. |
| LightGBM | [LightGBM](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html). |
| KurveRSC | [Original report](kurversc-technical-report.md); [current report](kurversc-staged-technical-report.md). |
| Constant predictors | [RelArena baseline definitions](https://arxiv.org/abs/2608.16319v2). |
