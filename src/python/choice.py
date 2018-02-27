"""Execute the specified method for random choice with arbitrary probability distribution and output information about execution

usage: 
    choice header
    choice (alias|alias-fast|binsearch-old|binsearch-fixed) <p_file> <sample_size>
"""

from docopt import docopt
from itertools import repeat, chain
import sys
import numpy as np
import alias as al
import time
from math import log

class StopWatch(object):

    def __init__(self):
        self.start = self.end = None

    def __enter__(self):
        self.start = time.perf_counter()
        self.end = None
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        return False

BUFFER_SIZE=( 1 << 19 ) # 4 MB

if __name__ == '__main__':
    args = docopt(__doc__)
    
    if args['header']:
        print('method,distribution_size,sample_size,init_time,time,median_error,mean_error,std_error,min_error,max_error')
        sys.exit(0)

    p = np.fromfile(args['<p_file>'])
    if np.any(np.isnan(p)):
        print('distribution contains nan, aborting', file=sys.stderr)
        sys.exit(1)

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
        with StopWatch() as sw:
            dist = np.random.pdist.wrap(p)
            dist.cumsum
        init_time = sw.elapsed

        def choice(k): 
            return np.random.choice(values, p=dist, size=k)

    elif args['alias']:
        method_name = 'alias'
        with StopWatch() as sw:
            table = al.aliastable(p)
        init_time = sw.elapsed

        def choice(k):
            return al.choice(table, values, k)

    elif args['alias-fast']:
        method_name = 'alias-fast'
        with StopWatch() as sw:
            table = al.aliastable(p)
        init_time = sw.elapsed

        def choice(k):
            return al.fast_choice(table, values, k)

    else:
        print('invalid args', file=sys.stderr)
        print('available options: alias, binsearch-old, binsearch-fixed', file=sys.stderr)
        sys.exit(1)

    elapsed = 0
    hits = np.zeros(n, dtype=int)
    total = 0
    current_size = 1

    while total < sample_size:
        ks = repeat( BUFFER_SIZE, current_size // BUFFER_SIZE )
        ks = chain( ks, (current_size % BUFFER_SIZE,) )
        for k in ks:
            with StopWatch() as sw:
                r = choice(k)

            total += k
            elapsed += sw.elapsed
            (ix, counts) = np.unique(r, return_counts=True)
            hits[ix] += counts

        freq = hits/np.sum(hits)
        error = (freq - p)
        nonzero_mask = (p != 0)
        error[nonzero_mask] = error[nonzero_mask]/p[nonzero_mask]
        error[(~nonzero_mask) & (freq != 0)] = float('inf')
        print(method_name
                , n
                , total
                , init_time
                , elapsed
                , np.median(error)
                , np.mean(error)
                , np.std(error)
                , np.min(error)
                , np.max(error), sep=',')

        current_size <<= 1
        if total + current_size > sample_size:
            current_size = sample_size - total
