'''Takes two np.ndarray objects and compute, if possible, the matrix product between them.

Your code will receive two variables, a and b. Compute their matrix product and assign it to out.

You are not allowed to use numpy.dot or numpy.matmul.'''

import numpy as np
import random
sha, shb, shc = random.randint(1, 4),  random.randint(1, 4), random.randint(1, 4)
a = np.random.random((sha, shb))
b = np.random.random((shb, shc))

#
if a.shape[1] == b.shape[0]:
    res = np.zeros([a.shape[0], b.shape[1]])
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            for k in range(b.shape[0]):
                res[i][j] += a[i][k] * b[k][j]
    print(res)
else:
    exit(1)
    print('Errore dimensione matrici')
