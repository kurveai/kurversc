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

The tables compare KurveRSC 0.2.0 (`kurversc-2026-10-22`) with the
latest TabPFN-Rel API and local variants (both dated September 28, 2026), RT-J,
RT-PluRel, Nori-Rel and GraphSAGE. This set covers strong aggregate competitors
and several relational modeling approaches; the sources below
provide their method citations. The short headers API and Local denote the
two TabPFN-Rel variants. All values are selected held-out test scores from the
same frozen RelArena file. **Bold marks the best score among the seven displayed
methods**, determined before rounding; it does not indicate statistical
significance. The complete comparison below includes all sixteen methods, including earlier
releases and simpler baselines.

| Dataset / task | KurveRSC | API | Local | RT-J | RT-PluRel | Nori-Rel | GraphSAGE |
|:--------------------------|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|
| Amazon / item-churn | 0.828712 | 0.828653 | 0.827685 | **0.832556** | 0.831014 | 0.822805 | 0.830527 |
| Amazon / user-churn | 0.712558 | 0.711698 | 0.706670 | **0.714327** | 0.713592 | 0.690798 | 0.704596 |
| Avito / user-clicks | 0.660062 | 0.691631 | **0.698621** | 0.671975 | 0.670345 | 0.636418 | 0.608674 |
| Avito / user-visits | **0.672515** | 0.672453 | 0.671587 | 0.667930 | 0.662691 | 0.662070 | 0.665758 |
| Event / user-ignore | 0.850888 | 0.784037 | 0.817646 | 0.821496 | 0.823280 | **0.859148** | 0.758728 |
| Event / user-repeat | 0.782152 | 0.760495 | 0.760628 | **0.786598** | 0.777340 | 0.781217 | 0.784626 |
| F1 / driver-dnf | 0.722017 | **0.727809** | 0.724794 | 0.588469 | 0.643268 | 0.715532 | 0.717235 |
| F1 / driver-top3 | **0.834154** | 0.791702 | 0.810632 | 0.797757 | 0.750144 | 0.793922 | 0.725975 |
| H&M / user-churn | **0.707174** | 0.706021 | 0.705416 | 0.706981 | 0.703784 | 0.694094 | 0.698525 |
| Stack / user-badge | 0.876020 | 0.883443 | 0.884633 | **0.893610** | 0.893398 | 0.840720 | 0.888748 |
| Stack / user-engagement | 0.903861 | 0.907662 | **0.907741** | 0.904240 | 0.895384 | 0.896160 | 0.905609 |
| Trial / study-outcome | 0.760476 | **0.764556** | 0.758230 | 0.711371 | 0.697735 | 0.742175 | 0.686232 |

## Regression: MAE, lower is better

MAE is in each task's native units; bold marks the best of the seven displayed methods.

| Dataset / task | KurveRSC | API | Local | RT-J | RT-PluRel | Nori-Rel | GraphSAGE |
|:--------------------------|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|
| Amazon / item-ltv | **42.3828** | 45.2705 | 45.3496 | 42.5622 | 42.8036 | 43.7084 | 49.2453 |
| Amazon / user-ltv | 13.9894 | 13.8771 | 13.8858 | **13.8515** | 13.8798 | 14.1412 | 14.4153 |
| Avito / ad-ctr | 0.033258 | **0.031258** | 0.031350 | 0.034591 | 0.036435 | 0.031766 | 0.038966 |
| Event / user-attendance | 0.234934 | 0.241652 | 0.238005 | 0.241660 | 0.241207 | **0.233820** | 0.245018 |
| F1 / driver-position | 4.0605 | **3.7574** | 3.7785 | 3.7911 | 3.8535 | 3.8844 | 4.0111 |
| H&M / item-sales | **0.033636** | 0.036479 | 0.036539 | 0.037345 | 0.039304 | 0.043806 | 0.055153 |
| Stack / post-votes | 0.062864 | **0.062733** | 0.062842 | 0.068049 | 0.064124 | 0.065047 | 0.064898 |
| Trial / site-success | 0.398749 | 0.377615 | 0.404219 | 0.362282 | 0.371274 | 0.417738 | **0.324851** |
| Trial / study-adverse | 33.1750 | 34.4014 | 35.5059 | 37.4920 | 34.0414 | **30.8832** | 44.3153 |

There are **11 wins and 10 losses** against the latest API client. These are
pairwise comparisons, not counts of best scores across all methods. Relative
to the earlier KurveRSC registration, 15 scores improve and 6 regress. These
are observed results at one seed, not a per-component ablation or a guarantee
of future improvement. See [measured resources](kurversc-staged-resources.md)
and the [technical report](kurversc-staged-technical-report.md).

## Complete sixteen-method comparison

All sixteen methods in the combined leaderboard are included below. Panels A
and B split columns for readability and follow the leaderboard order.
**Bold marks the best full-precision score across both panels together** for
each task. Consequently, a panel may have no bold entry on a particular row.
All scores use the selected seed-zero rows from the same frozen result file;
method citations are given in the sources below.

API-09 and Local-09 are the September 28, 2026 TabPFN-Rel releases; API-08 and
Local-08 are the August 15, 2026 releases. Kurve-02 is KurveRSC 0.2.0
(`kurversc-2026-10-22`); Kurve-01 is the earlier `kurversc-2026-09-03`
registration. RelGNN denotes `relgnn-es`; Entity and Global denote the
per-entity and global constant predictors. Scores are point estimates; bold
does not establish statistical significance.


### Classification: ROC-AUC, higher is better

Panel A.

| Dataset / task | API-09 | Kurve-02 | Local-09 | RT-J | API-08 | RT-PluRel | Kurve-01 | Nori-Rel |
|:--------------------------|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|
| Amazon / item-churn | 0.828653 | 0.828712 | 0.827685 | **0.832556** | 0.827996 | 0.831014 | 0.827256 | 0.822805 |
| Amazon / user-churn | 0.711698 | 0.712558 | 0.706670 | **0.714327** | 0.708649 | 0.713592 | 0.710347 | 0.690798 |
| Avito / user-clicks | 0.691631 | 0.660062 | **0.698621** | 0.671975 | 0.675191 | 0.670345 | 0.658406 | 0.636418 |
| Avito / user-visits | 0.672453 | 0.672515 | 0.671587 | 0.667930 | 0.668026 | 0.662691 | **0.674721** | 0.662070 |
| Event / user-ignore | 0.784037 | 0.850888 | 0.817646 | 0.821496 | **0.878659** | 0.823280 | 0.832628 | 0.859148 |
| Event / user-repeat | 0.760495 | 0.782152 | 0.760628 | **0.786598** | 0.759291 | 0.777340 | 0.779679 | 0.781217 |
| F1 / driver-dnf | 0.727809 | 0.722017 | 0.724794 | 0.588469 | 0.732172 | 0.643268 | **0.746484** | 0.715532 |
| F1 / driver-top3 | 0.791702 | **0.834154** | 0.810632 | 0.797757 | 0.771426 | 0.750144 | 0.664957 | 0.793922 |
| H&M / user-churn | 0.706021 | **0.707174** | 0.705416 | 0.706981 | 0.705215 | 0.703784 | 0.696916 | 0.694094 |
| Stack / user-badge | 0.883443 | 0.876020 | 0.884633 | **0.893610** | 0.880386 | 0.893398 | 0.877626 | 0.840720 |
| Stack / user-engagement | 0.907662 | 0.903861 | **0.907741** | 0.904240 | 0.905994 | 0.895384 | 0.903027 | 0.896160 |
| Trial / study-outcome | 0.764556 | 0.760476 | 0.758230 | 0.711371 | **0.764702** | 0.697735 | 0.712131 | 0.742175 |

Panel B.

| Dataset / task | Local-08 | GraphSAGE | RelGT | RDBLearn | RelGNN | LightGBM | Entity | Global |
|:--------------------------|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|
| Amazon / item-churn | 0.827857 | 0.830527 | 0.823835 | 0.819538 | 0.785579 | 0.662211 | 0.728875 | 0.500000 |
| Amazon / user-churn | 0.702403 | 0.704596 | 0.701924 | 0.684375 | 0.694281 | 0.517056 | 0.634205 | 0.500000 |
| Avito / user-clicks | 0.614522 | 0.608674 | 0.644353 | 0.678769 | 0.667571 | 0.564163 | 0.504143 | 0.500000 |
| Avito / user-visits | 0.668811 | 0.665758 | 0.662142 | 0.659624 | 0.648731 | 0.529282 | 0.602703 | 0.500000 |
| Event / user-ignore | 0.701376 | 0.758728 | 0.781507 | 0.664351 | 0.805393 | 0.777181 | 0.839930 | 0.500000 |
| Event / user-repeat | 0.769251 | 0.784626 | 0.734358 | 0.744084 | 0.754612 | 0.748295 | 0.751805 | 0.500000 |
| F1 / driver-dnf | 0.714468 | 0.717235 | 0.711667 | 0.714551 | 0.726106 | 0.730298 | 0.699258 | 0.500000 |
| F1 / driver-top3 | 0.792916 | 0.725975 | 0.810841 | 0.780081 | 0.758864 | 0.738889 | 0.556530 | 0.500000 |
| H&M / user-churn | 0.705690 | 0.698525 | 0.689531 | 0.698352 | 0.682025 | 0.590081 | 0.647972 | 0.500000 |
| Stack / user-badge | 0.863470 | 0.888748 | 0.574286 | 0.771147 | 0.620584 | 0.537995 | 0.788956 | 0.500000 |
| Stack / user-engagement | 0.905834 | 0.905609 | 0.906731 | 0.858670 | 0.905054 | 0.811836 | 0.826717 | 0.500000 |
| Trial / study-outcome | 0.730607 | 0.686232 | 0.668495 | 0.721205 | 0.657435 | 0.715018 | 0.500000 | 0.500000 |

### Regression: MAE, lower is better

Panel A.

| Dataset / task | API-09 | Kurve-02 | Local-09 | RT-J | API-08 | RT-PluRel | Kurve-01 | Nori-Rel |
|:--------------------------|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|
| Amazon / item-ltv | 45.2705 | 42.3828 | 45.3496 | 42.5622 | 46.7682 | 42.8036 | **42.2731** | 43.7084 |
| Amazon / user-ltv | 13.8771 | 13.9894 | 13.8858 | **13.8515** | 14.3582 | 13.8798 | 14.0205 | 14.1412 |
| Avito / ad-ctr | 0.031258 | 0.033258 | 0.031350 | 0.034591 | **0.031080** | 0.036435 | 0.033440 | 0.031766 |
| Event / user-attendance | 0.241652 | 0.234934 | 0.238005 | 0.241660 | 0.243944 | 0.241207 | 0.258668 | **0.233820** |
| F1 / driver-position | **3.7574** | 4.0605 | 3.7785 | 3.7911 | 3.7692 | 3.8535 | 3.8632 | 3.8844 |
| H&M / item-sales | 0.036479 | 0.033636 | 0.036539 | 0.037345 | 0.060507 | 0.039304 | **0.032168** | 0.043806 |
| Stack / post-votes | **0.062733** | 0.062864 | 0.062842 | 0.068049 | 0.067882 | 0.064124 | 0.063243 | 0.065047 |
| Trial / site-success | 0.377615 | 0.398749 | 0.404219 | 0.362282 | 0.412624 | 0.371274 | 0.403842 | 0.417738 |
| Trial / study-adverse | 34.4014 | 33.1750 | 35.5059 | 37.4920 | 39.7537 | 34.0414 | 41.2702 | **30.8832** |

Panel B.

| Dataset / task | Local-08 | GraphSAGE | RelGT | RDBLearn | RelGNN | LightGBM | Entity | Global |
|:--------------------------|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|
| Amazon / item-ltv | 47.7683 | 49.2453 | 48.6734 | 48.9975 | 52.4581 | 55.7508 | 65.3514 | 64.2336 |
| Amazon / user-ltv | 14.4009 | 14.4153 | 14.3521 | 14.5775 | 14.5752 | 16.7847 | 17.4235 | 16.7830 |
| Avito / ad-ctr | 0.031379 | 0.038966 | 0.036499 | 0.034103 | 0.042646 | 0.041250 | 0.041201 | 0.043067 |
| Event / user-attendance | 0.239383 | 0.245018 | 0.261493 | 0.242215 | 0.243858 | 0.262798 | 0.269152 | 0.263534 |
| F1 / driver-position | 3.7617 | 4.0111 | 4.7655 | 3.8886 | 4.2659 | 4.1058 | 4.1035 | 4.3991 |
| H&M / item-sales | 0.061362 | 0.055153 | 0.053168 | 0.067132 | 0.056493 | 0.075295 | 0.078033 | 0.076087 |
| Stack / post-votes | 0.067957 | 0.064898 | 0.067904 | 0.067719 | 0.067904 | 0.066099 | 0.069374 | 0.067904 |
| Trial / site-success | 0.385751 | **0.324851** | 0.370110 | 0.485833 | 0.340493 | 0.437506 | 0.441148 | 0.462222 |
| Trial / study-adverse | 42.5917 | 44.3153 | 44.1259 | 44.0267 | 46.2701 | 44.5735 | 57.5332 | 57.5332 |

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
