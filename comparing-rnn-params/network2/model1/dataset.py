import os
from time import process_time
import torch
import numpy as np
import pandas as pd
import h5py
from mappers import ClassMapper


class PerformanceEmbeddings(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_meta_csv_path: str,
            base_dir: str,
            class_mapper: ClassMapper
    ):

        # Read the metadata
        dataset = pd.read_csv(dataset_meta_csv_path)
        self.dataset = dataset.values.tolist()

        for row in self.dataset:
            [work_id, track_id] = row
            feature_path = [base_dir, work_id, "%s.%s" % (track_id, 'h5')]
            feature_path = os.path.join(*feature_path)
            with h5py.File(feature_path) as file:
                embedding = torch.from_numpy(file['embedding'][:])
                mapped_work_id = class_mapper.get_id(work_id)
                row.append(mapped_work_id)
                row.append(embedding)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        [work_id, track_id, mapped_work_id, embedding] = self.dataset[index]
        return (embedding, mapped_work_id)
