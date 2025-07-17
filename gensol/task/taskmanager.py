import os
import time
from typing import List, Optional
import pandas as pd
import numpy as np

from gensol.utils import local_shuffle

class TaskManager:
      
    _datadir: str
    _task_df: pd.DataFrame
    
    _u_types: np.ndarray[int]
    _overhead_matrix: np.ndarray[np.float32]

    _ids: np.ndarray[int]
    _types: np.ndarray[int]
    _lengths: np.ndarray[np.float32]
    _deadlines: np.ndarray[np.float32]
    _penalties: np.ndarray[np.float32]
    _harddeadlines: np.ndarray[np.float32]

    _solutions: np.ndarray[int]
    _partial_orderings: List[np.ndarray[int]]

    @property
    def n_types(self) -> int:
        return len(self._u_types)
    
    @property
    def solutions_folder(self) -> str:
        return os.path.join(self._datadir, "solutions")

    @property
    def solutions_filename(self) -> str:
        return os.path.join(self.solutions_folder, "solutions.npy")
    
    def __init__(
        self,
        datadir: str,
    ) -> None:
        self._datadir = datadir

        self._load_tasks(os.path.join(self._datadir, "tasks.csv"))
        self._load_oveheads(os.path.join(self._datadir, "overheads.csv"))

        self._init_solutions()
        self._init_partial_orderings()

    def _load_tasks(self, filename: str):
        self._task_df = pd.read_csv(filename, delimiter=',', skipinitialspace=True)
        self._init_tasks()
        for _, task in self._task_df.iterrows():
            self._add_task(*task)
        self._complete_tasks()
    
    def _init_tasks(self):
        self._ids_list           = []
        self._types_list         = []
        self._lengths_list       = []
        self._deadlines_list     = []
        self._penalties_list     = []
        self._harddeadlines_list = []

    def _add_task(
        self,
        id: int,
        type: int,
        length: np.float32,
        deadline: np.float32,
        penalty: np.float32,
        harddeadline: Optional[np.float32],
    ):
        self._ids_list.append(int(id))
        self._types_list.append(int(type))
        self._lengths_list.append(np.float32(length))
        self._deadlines_list.append(np.float32(deadline))
        self._penalties_list.append(np.float32(penalty))
        self._harddeadlines_list.append(np.inf if np.isnan(harddeadline) else np.float32(harddeadline))

    def _complete_tasks(self):
        self._ids = np.array(self._ids_list)
        sort_idcs = np.argsort(self._ids)
        self._ids = self._ids[sort_idcs]
        self._types = np.array(self._types_list)[sort_idcs]
        self._lengths = np.array(self._lengths_list)[sort_idcs]
        self._deadlines = np.array(self._deadlines_list)[sort_idcs]
        self._penalties = np.array(self._penalties_list)[sort_idcs]
        self._harddeadlines = np.array(self._harddeadlines_list)[sort_idcs]
    
    def _load_oveheads(self, filename: str):
        overhead_df = pd.read_csv(filename)
        self._u_types = np.union1d(np.unique(overhead_df["type_1"]), np.unique(overhead_df["type_2"]))
        
        # - Build Overhead Matrix - #
        self._overhead_matrix = np.zeros((self.n_types, self.n_types), dtype=np.float32)

        for _, oh_pair in overhead_df.iterrows():
            row_arg = np.argwhere(oh_pair["type_1"] == self._u_types)
            col_arg = np.argwhere(oh_pair["type_2"] == self._u_types)
            self._overhead_matrix[row_arg, col_arg] = oh_pair["oh"]
    
    def _init_solutions(self):
        
        solutions = []

        # - Order by deadline - #
        solutions.append(
            self._ids[np.argsort(self._deadlines)]
        )
        # - Order by time-to-complete - #
        solutions.append(
            self._ids[np.argsort(self._deadlines - self._lengths)]
        )

        self._solutions = np.stack(solutions, axis=0)

        # - Load previous solutions, if any - #
        if os.path.isfile(self.solutions_filename):
            self._solutions = np.concatenate([self._solutions, np.load(self.solutions_filename)], axis=0)
    
    def _init_partial_orderings(self):
        self._partial_orderings = []
        grouped = self._task_df.groupby(['type', 'penalty', 'length'])

        # Iterate over the groups
        for (key1, key2, key3), group in grouped:
            # Sort the group by column 'deadline'
            sorted_group = group.sort_values(by='deadline')
            
            # Filter out rows with not NaN values in column 'harddeadline'
            filtered_sorted_group = sorted_group[sorted_group['harddeadline'].isna()]
            
            # Extract the indices of the sorted rows after filtering and append to self._partial_orderings
            if len(filtered_sorted_group) > 1:
                self._partial_orderings.append(filtered_sorted_group.index.values)
    
    def _evaluate(self):
        lengths   = np.copy(self._lengths)[self._solutions]
        deadlines = self._deadlines[self._solutions]
        penalties = self._penalties[self._solutions]
        harddeadlines = self._harddeadlines[self._solutions]
        types     = self._types[self._solutions]
        overheads = self._overhead_matrix[np.searchsorted(self._u_types, types[:, :-1]), np.searchsorted(self._u_types, types[:, 1:])]

        lengths[:, 1:] += overheads
        completion_time = np.cumsum(lengths, axis=-1)
        penalty = np.maximum(np.zeros_like(completion_time), completion_time - deadlines) * penalties
        penalty += np.maximum(np.zeros_like(completion_time), completion_time - harddeadlines) * 9999
        
        score = np.sum(penalty, axis=-1) + completion_time[:, -1] / (completion_time.max() + 1)

        return score, penalty, completion_time
    
    def _solve(
        self,
        max_iterations: int = 1000,
        offspring_spawn: int = 100,
        shuffle_distance: int = 5,
        keep_n_best: int = 100,
        save_every: int = 100,
    ):
        for iteration in range(1, max_iterations+1):
            offspring = local_shuffle(
                np.tile(self._solutions, (offspring_spawn, 1)),
                distance=shuffle_distance,
                skip=1,
            )
            self._solutions = np.concatenate(
                [
                self._solutions,
                offspring,
                ],
                axis=0
            )
            score, _, _ = self._evaluate()
            best_score_idcs = np.argsort(score)[:keep_n_best]
            self._solutions = self._solutions[best_score_idcs]
            self.force_ordering()
            if iteration % save_every == 0:
                score, _, _ = self._evaluate()
                print(f"Iteration {iteration} - Best score: {np.min(score)}")
                os.makedirs(self.solutions_folder, exist_ok=True)
                np.save(self.solutions_filename, self._solutions)
        return self._evaluate()
    
    def force_ordering(self):
        for partial_ordering in self._partial_orderings:
            # Create a dictionary to map the values in B to their indices for easy comparison
            partial_ordering_index = {value: index for index, value in enumerate(partial_ordering)}

            def sort_according_to_partial_ordering(row, partial_ordering_index):
                # Separate elements present in B and those that are not
                in_po = [x for x in row if x in partial_ordering_index]
                not_in_po = [x for x in row if x not in partial_ordering_index]

                # Sort the elements that are present in partial_ordering_index
                in_po_sorted = sorted(in_po, key=lambda x: partial_ordering_index[x])

                # Merge sorted in_po and unsorted not_in_po maintaining their original positions
                sorted_row = []
                in_po_idx, not_in_po_idx = 0, 0

                for x in row:
                    if x in partial_ordering_index:
                        sorted_row.append(in_po_sorted[in_po_idx])
                        in_po_idx += 1
                    else:
                        sorted_row.append(not_in_po[not_in_po_idx])
                        not_in_po_idx += 1

                return sorted_row

            # Apply the sorting function to each row
            self._solutions = np.apply_along_axis(
                sort_according_to_partial_ordering,
                1,
                self._solutions,
                partial_ordering_index,
            )