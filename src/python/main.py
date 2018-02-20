import numpy as np
import alias as al
import time
import pandas as pd

np.random.seed(134)

m = (1 << 20)
n = (1 << 10) 
count = (1 << 11)

p = np.random.random(m)
#p = 1 + np.arange(m)
p = p/np.sum(p)

def timeit(f, k=None, kw=None): 
    if k is None:
        k = ()
    if kw is None:
        kw = {}

    start = time.perf_counter()
    result = f(*k, **kw)
    end = time.perf_counter()
    return (result, end - start)

def test_method(f, k, kw=None, count=1):
    total_time = 0
    hits = np.zeros(m, dtype='i')
    for i in range(count):
        (r, t) = timeit(f, k, kw)
        total_time += t
        (u, c) = np.unique(r, return_counts=True)
        hits[u] += c

    freq = hits/np.sum(hits)
    error = (freq - p) / p

    return (error, freq, total_time)


values = np.arange(m)

table = al.aliastable(p)

(np_error, np_freq, np_seconds) = test_method(np.random.choice, (values,), {'size':n, 'p':p}, count=count)
(al_error, al_freq, al_seconds) = test_method(al.choice, (table, values, n), count=count)

print('numpy time', np_seconds)
print('alias time', al_seconds)
alias_p = al.recover_distribution(table)
alias_error = (alias_p - p)/p

df = pd.DataFrame({ 'np': np_error, 'al': al_error, 'al_err': alias_error })

desc = df.describe()
print(desc)
print('proportional describe')
print(desc.al / desc.np)
al_maxerror = df.idxmax().al

print('expected p',  p[al_maxerror])
print('   alias p', alias_p[al_maxerror])
print('al freq p', al_freq[al_maxerror])
print('np freq p', np_freq[al_maxerror])
