import h5py
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing

infilename = 'data.h5'
datain = h5py.File(infilename, 'r')

# Visualization
# Walk vs Jump in hand
jd = datain['Lauren']['Jump']['jumping_inhandL3']
wd = datain['Lauren']['Walk']['walking_inhandL3']
for i in range(1, 5):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(jd[:, i], label='Jump')
    ax.plot(wd[:, i], label='Walk')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Acceleration (m/s^2)')
    ax.set_title('Walk vs Jump: {}'.format('X Axis' if (i == 1) else ('Y Axis' if (i == 2) else ('Z Axis' if (i == 3) else 'Absolute Acceleration'))))
    ax.legend()
    plt.show()

# Walk vs Jump in pocket
jd = datain['Jacob']['Jump']['Jumping20']
wd = datain['Jacob']['Walk']['Walking20']
for i in range(1, 5):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(jd[:, i], label='Jump')
    ax.plot(wd[:, i], label='Walk')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Acceleration (m/s^2)')
    ax.set_title('Walk vs Jump: {}'.format('X Axis' if (i == 1) else ('Y Axis' if (i == 2) else ('Z Axis' if (i == 3) else 'Absolute Acceleration'))))
    ax.legend()
    plt.show()


