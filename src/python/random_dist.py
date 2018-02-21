"""Generate a random discrete probability distribution of size <n> and save it to <file>.

usage:
    random_dist -h
    random_dist <n> <file>"""

from os.path import exists
import numpy as np
import docopt
from sys import stderr, exit

if __name__ == '__main__':
    args = docopt.docopt(__doc__)

    try:
        n = int(args['<n>'])
    except ValueError:
        n = 0

    if n <= 0:
        print('<n> must be a positive integer', file=stderr)
        exit(1)

    path = args['<file>']
    if exists(path):
        print('file "{}" exists, aborting.'.format(path), file=stderr)
        exit(1)

    p = np.random.random(n)
    p = p/np.sum(p)
    p.tofile(path)
