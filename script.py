import numpy as np
from gensol.task import TaskManager

tm = TaskManager("data/benchmark")

score, penalty, completion_time = tm._solve(
    max_iterations = 100000,
    offspring_spawn = 200,
    shuffle_distance = 5,
    keep_n_best = 200,
    save_every = 100,
)

np.save('solutions.npy', tm._solutions)
print(np.min(score))
print(tm._solutions[np.argmin(score)])