import os
import sys
import numpy as np

if __name__ == '__main__':
    root = sys.argv[1]

    summary = []
    for folder in sorted(os.listdir(root)):
        npz_file = os.path.join(root, folder, 'report_8.npz')
        if os.path.isfile(npz_file):
            npz = np.load(npz_file)
            summary_str = ""
            for k in npz.keys():
                summary_str += f'{k}: {npz[k]}\t'
            summary.append(f'{folder} \t | ' + summary_str)
    for s in summary:
        print(s)