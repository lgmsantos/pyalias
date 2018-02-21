"""Execute the specified method for random choice with arbitrary probability distribution and output information about execution

usage: 
    choice header
    choice (alias|binsearch-old|binsearch-fixed) <p_file> <sample_size>
"""

from docopt import docopt
import sys
import numpy as np
import alias as al
import time

BUFFER_SIZE=1000

if __name__ == '__main__':
    args = docopt(__doc__)
    
    if args['header']:
        print('method,distribution_size,sample_size,init_time,time,mean_error,std_error,min_error,max_error')
        sys.exit(0)

    p = np.fromfile(args['<p_file>'])
    n = len(p)
    values = np.arange(n)
    sample_size = int(args['<sample_size>'])

    if args['binsearch-old']:
        method_name = 'binsearch-old'
        init_time = 0
        def choice(k): 
            return np.random.choice(values, p=p, size=k)

    elif args['binsearch-fixed']:
        method_name = 'binsearch-fixed'
        start = time.perf_counter()
        dist = np.random.pdist.wrap(p)
        dist.cumsum
        end = time.perf_counter()
        init_time = end - start
        def choice(k): 
            return np.random.choice(values, p=dist, size=k)

    elif args['alias']:
        method_name = 'alias'
        start = time.perf_counter()
        table = al.aliastable(p)
        end = time.perf_counter()
        init_time = end - start
        def choice(k):
            return al.choice(table, values, k)
        
    else:
        print('invalid args', file=sys.stderr)
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
