# KurveRSC: maintainer reproduction resources

These are the successful jobs in the November 5, 2026 maintainer reproduction,
using `kurversc==0.2.0` and `graphreduce==1.10.18`, one task per CPU host.
[Immutable source](https://github.com/PriorLabs/relarena/blob/fcab8fb46b82299321c101b9a921d9a3f7c72263/docs/models/kurversc-staged-resources.md) ·
[Scores](kurversc-reproduced-results.md) ·
[Full-precision extract](assets/kurversc-relarena-reproduction-20261105.json).

## Host and engine settings

Eight tasks ran on 96-vCPU, 739-GB hosts with
`KURVERSC_DUCKDB_MEMORY_LIMIT=160GiB` and
`KURVERSC_DUCKDB_MAX_TEMP_DIRECTORY_SIZE=100GiB`.
Thirteen ran on 48-core, 190-GB hosts with `48GiB` and `20GiB`, respectively.
Host RAM is GB; Slurm `MaxRSS` is GiB. Peaks grow with engine limits and are
observations under these settings, not minimum-memory requirements.

## Per-task successful-job measurements

| Task | Host RAM, GB | Job wall time, h | Peak RSS, GiB |
|:---------------------------|------------:|----------------:|--------------:|
| Amazon / item-churn | 739 | 3.32 | 364 |
| Amazon / item-ltv | 739 | 6.48 | 202 |
| Amazon / user-churn | 739 | 4.98 | 262 |
| Amazon / user-ltv | 739 | 8.86 | 327 |
| Avito / ad-ctr | 190 | 0.71 | 34 |
| Avito / user-clicks | 190 | 1.21 | 51 |
| Avito / user-visits | 190 | 1.14 | 64 |
| Event / user-attendance | 190 | 3.35 | 46 |
| Event / user-ignore | 739 | 1.83 | 283 |
| Event / user-repeat | 190 | 1.72 | 44 |
| F1 / driver-dnf | 190 | 5.88 | 6 |
| F1 / driver-position | 190 | 4.88 | 4 |
| F1 / driver-top3 | 190 | 3.93 | 4 |
| H&M / item-sales | 190 | 3.47 | 27 |
| H&M / user-churn | 190 | 1.64 | 21 |
| Stack / post-votes | 739 | 10.60 | 257 |
| Stack / user-badge | 739 | 10.38 | 370 |
| Stack / user-engagement | 739 | 4.36 | 219 |
| Trial / site-success | 190 | 1.73 | 89 |
| Trial / study-adverse | 190 | 6.36 | 30 |
| Trial / study-outcome | 190 | 2.27 | 23 |

Successful job wall time spans **0.71-10.60 h**, and MaxRSS spans **4-370 GiB**.
The score CSV's separate application `time_total` spans 0.70-10.59 h and sums
to **88.81 task-hours**. This sum is not elapsed campaign time and excludes
failed earlier attempts; job and application clocks are reported separately.

## Retries and execution guidance

User-ignore and user-engagement ran out of memory on 190-GB hosts with the
48-GiB DuckDB limit; post-votes reached 140 GiB there before being moved.
All three completed on 739-GB hosts. This is a complete final reproduction,
not a claim of first-attempt success for every task.

DuckDB working-memory limits control spill behavior rather than total task RAM.
Slurm MaxRSS is not the earlier author-run cgroup peak or sampled process-tree
RSS; those values should not be substituted into this table. The published
record does not include every temporary-storage peak or utilization trace.

Concurrent tasks require separate output/scratch/cache directories and host
headroom. Preserve the procedure's internal worker/thread settings, and size
admission from measured demand and spill behavior. The absolute task deadline
remains below 24 h across resumes; incomplete execution is not a scored success.

The [October 27 resource record](archive/kurversc-staged-resources-20261027.md)
belongs to the superseded procedure.
