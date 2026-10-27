"""Run existing phase materialization with TRAIN/VAL-only source construction.

RelArena's usual constructor also loads masked test rows. This diagnostic must
not read, generate, or download any official task table, even a masked one.
The inherited inner split and all graph materialization code remain unchanged.
"""

from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relarena.dataset import RelBenchDatasetTask, drop_noncanonical_columns, drop_noncanonical_task_columns
from relarena.identity import relbench_run_identity


class ValidationOnlySource(RelBenchDatasetTask):
    def __init__(self, dataset_name, task_name):
        self.dataset_name, self.task_name = dataset_name, task_name
        self._identity = relbench_run_identity(dataset_name, task_name)
        self._dataset = get_dataset(dataset_name, download=False)
        self._task = get_task(dataset_name, task_name, download=False)
        self._db = drop_noncanonical_columns(self._dataset.get_db(), dataset_name)
        self._tables = {split:drop_noncanonical_task_columns(self._task, self._task.get_table(split), dataset_name)
                        for split in ("train", "val")}

    def outer_split(self):
        raise RuntimeError("Official inputs forbidden in validation-only materialization")


def main():
    from scripts import materialize_policy_folds as original
    original.RelBenchDatasetTask = ValidationOnlySource
    original.main()


if __name__ == "__main__":
    main()
