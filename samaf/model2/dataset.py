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
CACHE_LIMIT = 5000


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
            with h5py.File(feature_path) as file:
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
        return self.process_frames(np.array(frames))

    def process_frames(self, frames):
        # Get frames to [sequence_size, feature_size]
        frames = frames.transpose()

        maxIndices = np.argmax(frames, axis=1)

        filteredFrames = np.zeros(frames.shape, dtype=np.bool)
        for (step, index) in enumerate(maxIndices):
            filteredFrames[step, index] = 1.0

        # filteredFrames = filteredFrames[:, :-CQT_TOP_DROP_BINS]

        # [sequence_size,feature_size]
        X = torch.from_numpy(filteredFrames[:, :]).type(torch.float32)

        return X

    def get_performance(self, work_id: str, track_id: str):
        cache_key = "%s:%s" % (work_id, track_id)
        cache_result = self.cache.get(cache_key)
        if cache_result is not None:
            return cache_result

        performance_path = [self.base_dir, work_id, "%s.%s" % (track_id, 'h5')]
        performance_path = os.path.join(*performance_path)
        with h5py.File(performance_path) as file:
            data = np.copy(file[self.feature_type][:])

        self.cache.set(cache_key, data)
        return data
