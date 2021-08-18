import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment
import numpy as np
import os

import argparse
from time import time

from utils import listDir


DATASET = "basic"
CQT = "/Users/pasinduwijesena/Documents/university/research/experiments/data/{}_cqt_log/".format(DATASET)
WAV = "/Users/pasinduwijesena/Documents/university/research/experiments/data/{}".format(DATASET)
songsList = listDir(CQT, directoriesOnly=True)
songsList



startAt = 2 # 29
endAt = 3
progress = 0
for song in songsList:
    print("Song: ", song)
    if startAt > progress:
        progress = progress + 1
        print("\tSKIP")
        continue
    
    if endAt <= progress:
        print("STOP")
        break

    performances = listDir(os.path.join(CQT, song), filesOnly=True)
    for performanceCQT in performances:
        performance = '.'.join(performanceCQT.split('.')[:-1])
        performanceWAV = "{}.wav".format(performance)

        waveform, sr = librosa.load(os.path.join(WAV, song, performanceWAV))
        cqt = np.load(os.path.join(CQT, song, performanceCQT))

        print(performance)
        print("Waveform: {}, at {}Hz. {}s".format(waveform.shape, sr, waveform.shape[0]/sr))

        plt.figure()
        librosa.display.specshow(cqt, x_axis="time")
        plt.title(performance)
        print("\n\n")

    plt.show()
    progress = progress + 1
    print("{}/{}".format(progress, len(songsList)))
    break