import sys
import numpy as np
import alias as al
import time

BUFFER_SIZE=1000

if __name__ == '__main__':
    
    args = sys.argv[1:]
    method_name = args[0]

    if method_name == 'header':
        print('method,distribution_size,sample_size,init_time,time,mean_error,std_error,min_error,max_error')
        sys.exit(0)

    p_path = args[1]
    p = np.fromfile(p_path)
    n = len(p)
    values = np.arange(n)
    sample_size = int(args[2])

    if method_name == 'binsearch-old':
        init_time = 0
        def choice(k): 
            return np.random.choice(values, p=p, size=k)

    elif method_name == 'binsearch-fixed':
        start = time.perf_counter()
        dist = np.random.pdist.wrap(p)
        dist.cumsum
        end = time.perf_counter()
        init_time = end - start
        def choice(k): 
            return np.random.choice(values, p=dist, size=k)

    elif method_name == 'alias':
        start = time.perf_counter()
        table = al.aliastable(p)
        end = time.perf_counter()
        init_time = end - start
        def choice(k):
            return al.choice(table, values, k)
        
    else:
        print('invalid method:', method_name, file=sys.stderr)
        print('available options: alias, binsearch-old, binsearch-fixed', file=sys.stderr)
        sys.exit(1)

    elapsed = 0
    hits = np.zeros(n, dtype='i')
    total = 0
    while total < sample_size:
        if total + BUFFER_SIZE > sample_size:
            size = sample_size - total
        else:
            size = BUFFER_SIZE

        start = time.perf_counter()
        r = choice(size)
        end = time.perf_counter()

        total += size
        elapsed += end - start
        (ix, counts) = np.unique(r, return_counts=True)
        hits[ix] += counts

    freq = hits/np.sum(hits)
    error = (p - freq)/p
    print(method_name, n, total,
            init_time, elapsed, np.mean(error),
            np.std(error), np.min(error), np.max(error), sep=',')
