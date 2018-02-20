import numpy as np
cimport numpy as np

def aliastable(np.ndarray p):

    cdef np.ndarray GS, F, L
    cdef int n, i, j, k, s, g

    n = len(p)

    GS = np.empty(n, dtype=np.int)
    (g, s) = (-1, n)
    F = n * p
    L = np.arange(n, dtype=np.int)

    for i in range(n):
        if F[i] >= 1.:
            g += 1
            GS[g] = i
        else:
            s -= 1
            GS[s] = i
            
    for i in range(n-1):
        k = GS[g]
        j = GS[s]
        L[j] = k
        F[k] = F[k] - (1.0 - F[j])

        if F[k] < 1:
            g -= 1
            GS[s] = k
        else:
            s += 1

    return (F, L)

def recover_distribution(table):
    (F, L) = table
    n = len(F)
    p = np.zeros(n, dtype=F.dtype)
    cdef size_t i
    for i in range(n):
        p[i] += F[i]/n
        p[L[i]] += (1 - F[i])/n
    return p


def choice(table, values, n):
    (F, L) = table
    m = len(values)
    ix = (m * np.random.random(n)).astype(np.int)
    mask = F[ix] < np.random.random(n)
    ix[mask] = L[ix[mask]] 
    return values[ix]
