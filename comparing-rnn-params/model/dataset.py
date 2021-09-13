import os
import torch
import numpy as np
import librosa as la
from utils import listDir, tick, tock

FRAMES_PER_SAMPLE: int = 336  # number of frames per sample
HOP_LENGTH: int = 42  # number of frames to hop, to get to next sample
# number of samples to extract from a performance
SAMPLES_PER_PERFORMANCE: int = 120

# CQT Filtering Params
CQT_TOP_DROP_BINS: int = 36
CQT_PRESERVED_PEAK_COUNT: int = 1

# Audio data Cache limits
CACHE_LIMIT = 80


class Covers80DatasetPerformanceChunks(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, excluded_transforms: list = [], validation: bool = False, isolated_performance_index=None):
        performances = getPerformancesList(
            root_dir=root_dir, excluded_transforms=excluded_transforms)
        self.performances = []
        self.cache = {}
        self.cache_size = 0
        self.isolated_performance_index = isolated_performance_index

        for performance in performances:
            self.performances.append(performance['path'])

        if isolated_performance_index != None:
            print("Using only {}".format(self.performances[isolated_performance_index]))

    def __len__(self):
        if self.isolated_performance_index != None:
            return 1 * SAMPLES_PER_PERFORMANCE
        return len(self.performances) * SAMPLES_PER_PERFORMANCE

    def __getitem__(self, index):
        tick("GET ITEM {}".format(index))
        performance_index = index // SAMPLES_PER_PERFORMANCE
        if self.isolated_performance_index != None:
            performance_index += self.isolated_performance_index * SAMPLES_PER_PERFORMANCE

        if self.performances[performance_index] in self.cache:
            cqt = self.cache[self.performances[performance_index]]
        else:
            # Free one item  from the cache if the cache limit has reached
            if self.cache_size > CACHE_LIMIT:
                cache_keys = list(dict.keys())
                del self.cache[cache_keys[0]]

            cqt = np.load(self.performances[performance_index])
            self.cache[self.performances[performance_index]] = cqt

        frame_offset = (index % SAMPLES_PER_PERFORMANCE) * HOP_LENGTH
        # [feature_size, sequence_size]
        frames = cqt[:, frame_offset:(frame_offset+FRAMES_PER_SAMPLE)]
        frames = frames.transpose()  # [sequence_size, feature_size]

        frames[:, -CQT_TOP_DROP_BINS:] = 0.0
        maxIndices = np.argmax(frames, axis=1)

        filteredFrames = np.zeros(frames.shape, dtype=np.bool)
        for (step, index) in enumerate(maxIndices):
            filteredFrames[step, index] = 1.0

        filteredFrames = filteredFrames[:, :-CQT_TOP_DROP_BINS]

        # [sequence_size,feature_size]
        X = torch.from_numpy(filteredFrames[:-1, :]).type(torch.float32)
        # [sequence_size]
        Y = torch.as_tensor(np.argmax(filteredFrames[-1, :]))
        tock("GET ITEM {}".format(index))

        return X, Y


def getPerformancesList(root_dir: str, excluded_transforms: list = []):
    songs = listDir(path=root_dir, directoriesOnly=True)
    all_performances = []
    for song in songs:
        song_dir = os.path.join(root_dir, song)
        performances = listDir(song_dir, filesOnly=True)
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
                "song": song,
                "name": name,
                "path": os.path.join(song_dir, performance)
            }
            all_performances.append(data)

    return all_performances