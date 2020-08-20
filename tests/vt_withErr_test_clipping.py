import random
from timeit import Timer

import numpy as np

import fastvector

print("Hallo Welt")


v = fastvector.VectorND([random.random() for _ in range(100_000)])
a = np.array([random.random() for _ in range(100_000)])

# Timing test
import_string = \
'''from __main__ import v, a
import fastvector
import numpy as np'''  

python_timer = Timer(
    'fastvector.python_clip_vector(v, -1, 1, v)', 
    setup=import_string)

naive_cython_timer = Timer(
    'fastvector.naive_cython_clip_vector(v, -1, 1, v)', 
    setup=import_string)

cython_timer = Timer(
    'fastvector.cython_clip_vector(v, -1, 1, v)', 
    setup=import_string)

numpy_timer = Timer(
    'np.clip(a, -1, 1, a)', 
    setup=import_string)

num_runs = 100
print('fastvector.python_clip_vector')
print(sum(python_timer.repeat(number=num_runs)) / num_runs)
print('fastvector.naive_cython_clip_vector')
print(sum(naive_cython_timer.repeat(number=num_runs)) / num_runs)
print('fastvector.cython_clip_vector')
print(sum(cython_timer.repeat(number=num_runs)) / num_runs)
print('np.clip')
print(sum(numpy_timer.repeat(number=num_runs)) / num_runs)
