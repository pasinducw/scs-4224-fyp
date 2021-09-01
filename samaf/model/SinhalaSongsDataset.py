import os
import math
import torch
import numpy as np


class SinhalaSongsDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, trim_seconds=10, test=False, validation=False):
        self.dir = root_dir
        self.feature_list = sorted(os.listdir(self.dir))
        self.trim_seconds = trim_seconds

        length = len(self.feature_list)
        block_1 = math.floor((length/5.0) * 0.6) * 5  # 60% of originals
        block_2 = math.floor((length/5.0) * 0.8) * 5  # 80% of originals

        start_index = 0
        end_index = block_1

        if test == True:
            start_index = block_1
            end_index = block_2
        if validation == True:
            start_index = block_2
            end_index = length

        self.feature_list = self.feature_list[start_index:end_index]

    def __len__(self):
        return len(self.feature_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        song_id = self.feature_list[idx].split(".")[0]

        song_path = os.path.join(self.dir, self.feature_list[idx])
        _, features = np.load(song_path, allow_pickle=True)
        # mfccs = torch.from_numpy(np.load(song_path))

        trim_frames = self.trim_seconds * 100
        # mfccs = mfccs[:, :trim_frames]  # trim song to given number of seconds

        all_mfccs = []

        for mfccs in features:
            # trim song to given number of seconds
            mfccs = torch.from_numpy(mfccs[:, :trim_frames])

            # converting to shape [M_number_of_mfcc_coefficients, I_MFCC_blocks, T_number_of_steps]
            mfccs = mfccs.view(13, -1, 100)
            # converting to shape [I_MFCC_blocks, T_number_of_steps, M_number_of_mfcc_coefficients]
            mfccs = mfccs.permute(1, 2, 0)
            all_mfccs.append(mfccs)

        return int(song_id), torch.cat(all_mfccs).float()
