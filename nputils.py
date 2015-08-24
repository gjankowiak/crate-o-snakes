import numpy as np

def blkdiag(matrices):
    matrices = filter(lambda x:x.size, matrices)

    shapes = []

    for mat in matrices:
        shapes.append(list(mat.shape))

    shapes = np.array(shapes)

    M_shape = np.sum(shapes, axis=0)

    M = np.zeros(M_shape)

    [n, m] = [0, 0]
    for i, s in enumerate(shapes):
        M[n:n+s[0], m:m+s[1]] = matrices[i]
        [n, m] = [n+s[0], m+s[1]]
