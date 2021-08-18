import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment
import numpy as np

import argparse
from time import time
from IPython.display import Audio

from utils import listDir

import sox




%load_ext autoreload
%autoreload 2
%reload_ext autoreload







CQT = "/home/pasinducw/Downloads/Research-Datasets/covers80_cqt/"
WAV = "/home/pasinducw/Downloads/Research-Datasets/covers80/"
songsList = listDir(CQT, directoriesOnly=True)
songsList





progress =0
for song in songsList:
    performances = listDir(os.path.join(CQT, song), filesOnly=True)
    for performanceCQT in performances:
        performance = '.'.join(performanceCQT.split('.')[:-1])
        performanceWAV = "{}.wav".format(performance)

        waveform, sr = librosa.load(os.path.join(WAV, song, performanceWAV))
        cqt = np.load(os.path.join(CQT, song, performanceCQT))

        print(performance)
        print("Waveform: {}, at {}Hz. {}s".format(waveform.shape, sr, waveform.shape[0]/sr))
        Audio(waveform, rate=sr)
        librosa.display.specshow(cqt, x_axis="time")
        plt.show()

        print("\n\n")
        break

    progress = progress + 1
    print("{}/{}".format(progress, len(songsList)))