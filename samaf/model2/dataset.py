import os
from time import process_time
import torch
import numpy as np
import pandas as pd
import h5py
import math
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


class Performances(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_meta_csv_path: str,
    ):
        dataset = pd.read_csv(dataset_meta_csv_path, dtype=str)
        self.dataset = dataset.values.tolist()
        pass

    def len(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset(index)


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
            include_augmentations: bool = False,
            augmentations_base_dir: str = None,
            augmentations: list = None,
    ):

        # Read the metadata
        dataset = pd.read_csv(dataset_meta_csv_path, dtype=str)
        if work_id and track_id:
            dataset = (dataset[dataset['work_id'] == work_id])
            dataset = (dataset[dataset['track_id'] == track_id])
            self.dataset = dataset.values.tolist()
        else:
            self.dataset = dataset.values.tolist()

        # Store the number of frames of each performance in the dataset
        for row in self.dataset:
            [work_id, track_id] = row

            sequence_path = os.path.join(
                base_dir, work_id, "%s.%s" % (track_id, 'h5'))  # original
            seq_length = self.get_sequence_length(
                sequence_path, feature_type, time_axis)

            if include_augmentations:
                # Go through each variation, and pick the min sequence length
                for augmentation in augmentations:
                    sequence_path = os.path.join(
                        augmentations_base_dir, augmentation, work_id, "%s.%s" % (track_id, 'h5'))
                    seq_length = min(
                        seq_length,
                        self.get_sequence_length(
                            sequence_path, feature_type, time_axis)
                    )
            row.append(seq_length)

        # Compute the total sample count and store it
        self.samples = 0
        for row in self.dataset:
            samples = (row[2] - (frames_per_sample -
                                 hop_length)) // hop_length
            # ignore 10% of samples to help account for variation in number of frames between augmentations
            samples = math.floor(samples * 0.9)
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

        self.include_augmentations = include_augmentations
        self.augmentations_base_dir = augmentations_base_dir
        self.augmentations = augmentations

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

        variants = [None]
        if self.include_augmentations:
            variants = [None, *self.augmentations]

        # sequence, max_indices, work_id, track_id, index
        results = ([], [], work_id, track_id, index)

        for variant in variants:
            frames = self.get_performance(
                work_id=work_id, track_id=track_id, augmentation=variant)

            # Adjust the number of frames fetched and the frame index to account for variations that shrink or lengthen the track
            variant_frame_count = frames.shape[self.time_axis]
            scale_factor = variant_frame_count/frame_count
            frame_start_index = math.floor(
                sample_index * self.hop_length * scale_factor)
            frame_end_index = frame_start_index + self.frames_per_sample

            if self.time_axis == 1:
                frames = frames[:, frame_start_index: frame_end_index]
            else:
                frames = frames[frame_start_index: frame_end_index, :]
                frames = frames.transpose()

            # Prepare the extracted frames for the classification task
            X, max_indices = self.process_frames(np.array(frames))
            results[0].append(X)
            results[1].append(max_indices)

        results = (
            np.array(results[0], dtype=float),
            np.array(results[1], dtype=int),
            work_id,
            track_id,
            index,
        )

        if self.include_augmentations == False:
            return results[0][0], results[1][0], results[2], results[3]

        return (
            torch.from_numpy(results[0]).type(torch.float32),
            torch.from_numpy(results[1]).type(torch.long),
            results[2],
            results[3],
            results[4],
        )

    def process_frames(self, frames):
        # Get frames to [sequence_size, feature_size]
        frames = frames.transpose()

        maxIndices = np.argmax(frames, axis=1)

        filteredFrames = np.zeros(frames.shape, dtype=np.bool)
        for (step, index) in enumerate(maxIndices):
            filteredFrames[step, index] = 1.0

        # filteredFrames = filteredFrames[:, :-CQT_TOP_DROP_BINS]

        # [sequence_size,feature_size]
        # torch.from_numpy(filteredFrames[:, :]).type(torch.float32)
        X = filteredFrames[:, :]

        return (X, maxIndices)

    def get_performance(self, work_id: str, track_id: str, augmentation: str = None):
        cache_key = "%s:%s:%s" % (
            work_id, track_id, augmentation if augmentation != None else "original")
        cache_result = self.cache.get(cache_key)
        if cache_result is not None:
            return cache_result

        base_dir = os.path.join(self.augmentations_base_dir,
                                augmentation) if augmentation != None else self.base_dir

        performance_path = [base_dir, work_id, "%s.%s" % (track_id, 'h5')]
        performance_path = os.path.join(*performance_path)
        with h5py.File(performance_path) as file:
            data = np.copy(file[self.feature_type][:])

        self.cache.set(cache_key, data)
        return data

    def get_sequence_length(self, sequence_path: list, feature_type: str, time_axis: int):
        with h5py.File(sequence_path) as file:
            dimensions = file[feature_type].shape
        return dimensions[time_axis]
