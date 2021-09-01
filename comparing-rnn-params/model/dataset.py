import os
import torch
import numpy as np
import librosa as la
from utils import listDir, tick, tock

FRAMES_PER_SAMPLE: int = 168  # number of frames per sample
HOP_LENGTH: int = 42  # number of frames to hop, to get to next sample
# number of samples to extract from a performance
SAMPLES_PER_PERFORMANCE: int = 60

# CQT Filtering Params
CQT_TOP_DROP_BINS: int = 36
CQT_PRESERVED_PEAK_COUNT: int = 10


class Covers80DatasetPerformanceChunks(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, excluded_transforms: list = [], validation:bool=False):
        songs = getSongsMap(
            root_dir=root_dir, excluded_transforms=excluded_transforms)
        self.performances = []
        for song in songs:
            performances = songs[song]
            for performance in performances:
                self.performances.append(performance['path'])

    def __len__(self):
        return len(self.performances) * SAMPLES_PER_PERFORMANCE

    def __getitem__(self, index):
        tick("GET ITEM {}".format(index))
        performance_index = index // SAMPLES_PER_PERFORMANCE
        cqt = np.load(self.performances[performance_index])

        frame_offset = (index % SAMPLES_PER_PERFORMANCE) * HOP_LENGTH
        frames = cqt[:, frame_offset:(frame_offset+FRAMES_PER_SAMPLE)]

        frames[-CQT_TOP_DROP_BINS:, :] = 0.0
        framesLog = la.amplitude_to_db(np.abs(frames), ref=np.max)

        sortedPeaks = np.argsort(framesLog, axis=0)
        for (step, sortedPeak) in enumerate(np.transpose(sortedPeaks)):
            for i in sortedPeak[:-CQT_PRESERVED_PEAK_COUNT]:
                frames[i, step] = 0.0

        framesLog = la.amplitude_to_db(np.abs(frames), ref=np.max)
        framesLog = framesLog[:-CQT_TOP_DROP_BINS, :]
        framesLog = np.transpose(framesLog)  # turns to [sequence_size, feature_size]
        X = torch.from_numpy(framesLog[:-1, :])  # [sequence_size,feature_size]
        Y = torch.from_numpy(framesLog[-1, :])  # [feature]
        tock("GET ITEM {}".format(index))
        return X, Y


def getSongsMap(root_dir: str, excluded_transforms: list = []):
    songs = listDir(path=root_dir, directoriesOnly=True)
    songsMap = {}

    for song in songs:
        song_dir = os.path.join(root_dir, song)
        performances = listDir(song_dir, filesOnly=True)
        song_performances: list = []
        for performance in performances:
            name = '.'.join(performance.split('.')[:-1])

            # Check if excluded
            excluded = False
            for suffix in excluded_transforms:
                if name[len(name)-len(suffix):] == suffix:
                    excluded = True

            if excluded:
                continue

            data = {
                "name": name,
                "path": os.path.join(song_dir, performance)
            }
            song_performances.append(data)

        songsMap[song] = song_performances

    return songsMap
