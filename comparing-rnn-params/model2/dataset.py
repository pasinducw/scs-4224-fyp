import os
from time import process_time
import torch
import numpy as np
import pandas as pd
import h5py
from cache import SimpleCache
from utils import upper_bound

FRAMES_PER_SAMPLE: int = 336  # number of frames per sample
HOP_LENGTH: int = 2  # number of frames to hop, to get to next sample
# number of samples to extract from a performance
SAMPLES_PER_PERFORMANCE: int = 120

# CQT Filtering Params
CQT_TOP_DROP_BINS: int = 36
CQT_PRESERVED_PEAK_COUNT: int = 1

# Audio data Cache limits
CACHE_LIMIT = 80


class PerformanceChunks(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_meta_csv_path: str,
            base_dir: str,
            feature_type: str = 'cqt',
            time_axis: int = 1,
            work_id: str = None,
            track_id: str = None,
            cache_limit: int = CACHE_LIMIT,
            hop_length: int = HOP_LENGTH,
            frames_per_sample: int = FRAMES_PER_SAMPLE,
            drop_from_front: int = 0, # number of bins to drop from the front of a frame
            drop_from_end: int = 0, # number of bins to drop from the end of a frame
    ):

        # Read the metadata
        dataset = pd.read_csv(dataset_meta_csv_path)
        if work_id and track_id:
            dataset = (dataset[dataset['work_id'] == work_id])
            dataset = (dataset[dataset['track_id'] == track_id])
            self.dataset = dataset.values.tolist()
        else:
            self.dataset = dataset.values.tolist()

        # Store the number of frames of each performance in the dataset
        for row in self.dataset:
            [work_id, track_id] = row
            feature_path = [base_dir, work_id, "%s.%s" % (track_id, 'h5')]
            feature_path = os.path.join(*feature_path)
            file = h5py.File(feature_path)
            dimensions = file[feature_type].shape
            row.append(dimensions[time_axis])

        # Compute the total sample count and store it
        self.samples = 0
        for row in self.dataset:
            samples = (row[2] - (frames_per_sample -
                                 hop_length)) // hop_length
            self.samples += samples
            row.append(samples)  # number of samples in the performance
            row.append(self.samples)  # summation array formation

        # Setup the instance variables
        self.base_dir = base_dir
        self.feature_type = feature_type
        self.time_axis = time_axis
        self.hop_length = hop_length
        self.frames_per_sample = frames_per_sample
        self.cache = SimpleCache(cache_limit)
        self.drop_from_front = drop_from_front
        self.drop_from_end = drop_from_end

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        performance_index = upper_bound(
            self.dataset, (lambda row: row[4]), index)
        if performance_index > len(self.dataset):
            raise Exception('Request sample is out of range')

        [work_id, track_id, frame_count, sample_count,
            sample_summation_count] = self.dataset[performance_index]
        sample_index = sample_count - (sample_summation_count - index)

        frames = self.get_performance(work_id=work_id, track_id=track_id)
        if self.time_axis == 1:
            frames = frames[:, sample_index * self.hop_length: (
                sample_index * self.hop_length + self.frames_per_sample)]
        else:
            frames = frames[sample_index * self.hop_length: (
                sample_index * self.hop_length + self.frames_per_sample), :]
            frames = frames.transpose()

        # Prepare the extracted frames for the classification task
        return self.process_frames(np.array(frames), drop_from_end=self.drop_from_end, drop_from_front=self.drop_from_front)

    def process_frames(self, frames, drop_from_end=0, drop_from_front=0):
        # Get frames to [sequence_size, feature_size]
        frames = frames.transpose()

        if drop_from_end:
            frames = frames[:, :-drop_from_end]
        if drop_from_front:
            frames = frames[:, drop_from_front:]
        
        maxIndices = np.argmax(frames, axis=1)

        filteredFrames = np.zeros(frames.shape, dtype=np.bool)
        for (step, index) in enumerate(maxIndices):
            filteredFrames[step, index] = 1.0

        # [sequence_size,feature_size]
        X = torch.from_numpy(filteredFrames[:-1, :]).type(torch.float32)
        # [sequence_size]
        Y = torch.as_tensor(np.argmax(filteredFrames[-1, :]))

        return X, Y

    def get_performance(self, work_id: str, track_id: str):
        cache_key = "%s:%s" % (work_id, track_id)
        cache_result = self.cache.get(cache_key)
        if cache_result:
            return cache_result

        performance_path = [self.base_dir, work_id, "%s.%s" % (track_id, 'h5')]
        performance_path = os.path.join(*performance_path)
        file = h5py.File(performance_path)

        self.cache.set(cache_key, file[self.feature_type])
        return file[self.feature_type]


# base = '/Users/pasinduwijesena/Documents/university/research/src/data/acoss/covers80_features/'
# feature_type = 'cqt'
# time_axis = 1
# feature_axis = 0
# dataset = pd.read_csv(
#     '/Users/pasinduwijesena/Documents/university/research/src/data/acoss/covers80_features/annotations.csv')
# dataset = dataset[dataset['work_id'] == "Claudette"]
# dataset = dataset[dataset['track_id'] ==
#                   "everly_brothers+The_Fabulous_Style_of+01-Claudette"]
# print(dataset.head())
# dataset = dataset.values.tolist()
# for row in dataset:

#     [work_id, track_id] = row
#     feature_path = [base, work_id, "%s.%s" % (track_id, 'h5')]
#     feature_path = os.path.join(*feature_path)
#     file = h5py.File(feature_path)
#     print(file[feature_type][:, :].shape)
#     v = np.array(file[feature_type][:, :])
#     print(v.shape)
#     break

# dataset = PerformanceChunks(
#     dataset_meta_csv_path="/Users/pasinduwijesena/Documents/university/research/src/data/acoss/covers80_features/annotations.csv",
#     base_dir="/Users/pasinduwijesena/Documents/university/research/src/data/acoss/covers80_features/",
#     feature_type="cqt",
#     time_axis=1,
#     # work_id="Claudette",
#     # track_id="everly_brothers+The_Fabulous_Style_of+01-Claudette",
# )
# dataloader = torch.utils.data.DataLoader(
#     dataset, batch_size=10, num_workers=0, shuffle=False)

# print(len(dataset))
# for i, (X,Y) in enumerate(dataloader):
#     print("I is ", i, X.shape, Y)
