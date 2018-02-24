"""Generate a random discrete probability distribution of size <n> and save it to <file>.

usage:
    pdist -h
    pdist (random|linear|quad|exp) <n> <file>"""

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

    if args['random']:
        p = np.random.random(n)

    elif args['linear']:
        p = np.arange(n, dtype='d')

    elif args['quad']:
        p = np.arange(n, dtype='d') ** 2

    elif args['exp']:
        p = 2 ** np.arange(n, dtype='d')

    else:
        print('illegal subcommand', file=sys.stderr)
        sys.exit(1)

    np.random.shuffle(p)
    p = p/np.sum(p)
    p.tofile(path)
