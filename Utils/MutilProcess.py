import numpy as np
import pandas as pd
import tqdm, time
from multiprocessing import Process, Pool, Manager, Lock


def test(kwargs):
    time.sleep(0.5)
    return kwargs['1']


# must only use kwargs
def mutil_process(function, kwargs_ls, n_process=8):
    with Pool(processes=n_process) as pool:
        results = []
        # Use tqdm to create a progress bar
        with tqdm.tqdm(total=len(kwargs_ls), desc='RUN {} {} times with {} processes'.format(function.__name__, len(kwargs_ls), n_process), ncols=150) as pbar:
            for result in pool.imap_unordered(function, kwargs_ls):
                results.append(result)
                pbar.update()


if __name__ == '__main__':
    mutil_process(test, [{str(1): i} for i in range(100)], 10)
