import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')
plt.grid()

pd.set_option('display.width', None)

if __name__ == '__main__':

    files = sys.argv[1:]
    df = None
    for f in files:
        d = pd.read_csv(f)
        if df is None:
            df = d
        else:
            df = df.append(d, ignore_index=True)
