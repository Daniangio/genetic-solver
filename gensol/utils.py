import numpy as np


def local_shuffle(arr: np.ndarray, distance: int = 5, skip: int = 1):
    rows, cols = arr.shape
    idcs_array = np.arange(cols)
    idcs_array = np.tile(idcs_array, (rows, 1))

    shuffled_arr = np.copy(arr)
    for i in range(0, cols, skip):
        # Define the range within which the element can be swapped
        swap_range_start = max(0, i - distance)
        swap_range_end = min(cols - 1, i + distance)
        
        # Choose a random index within the swap range
        j = np.random.randint(swap_range_start, swap_range_end + 1, size=rows)
        
        # Swap elements
        temp_val = shuffled_arr[np.arange(rows), i]
        shuffled_arr[:, i] = shuffled_arr[np.arange(rows), j]
        shuffled_arr[np.arange(rows), j] = temp_val
            
    return shuffled_arr