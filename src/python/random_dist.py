from os.path import exists
import numpy as np
import sys 

def print_usage():
    print('random_dist <N> <PATH>')
    print('\tgenerate a random discrete probabilitiy distribution with <N> elements and save it to <PATH>')

if __name__ == '__main__':

    args = sys.argv[1:]
    if len(args) != 2:
        print_usage()
        sys.exit(1)

    n = int(args[0])
    path = args[1]

    if exists(path):
        print('file {} exists. aborting.', file=sys.stderr)
        sys.exit(1)

    p = np.random.random(n)
    p = p/np.sum(p)
    p.tofile(path)
