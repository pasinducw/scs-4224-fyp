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
            class_mapper: ClassMapper,
            mean: float = None,
            norm: float = None,
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

        # Drop the mean
        if mean is None:
            mean = np.mean([embedding.numpy()
                           for (_, _, _, embedding) in self.dataset], axis=0)
        for row in self.dataset:
            row[3] = (row[3] - mean)

        # Drop the variance
        if norm is None:
            norm = np.linalg.norm([embedding.numpy() for (
                _, _, _, embedding) in self.dataset], axis=0)
        for row in self.dataset:
            row[3] = (row[3]/norm)

        self.mean = mean
        self.norm = norm

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        [work_id, track_id, mapped_work_id, embedding] = self.dataset[index]
        return (embedding, mapped_work_id)


# mapper = ClassMapper()
# dataset1 = PerformanceEmbeddings(dataset_meta_csv_path="/home/pasinducw/Downloads/Research-Datasets/covers80/old/embeddings/metadata.csv",
#                                  base_dir="/home/pasinducw/Downloads/Research-Datasets/covers80/old/embeddings", class_mapper=mapper)


# print("-----------------------------------")
# an_arr = np.random.rand(1000)*10
# # print(an_arr)

# mean = np.mean(an_arr)
# print("Mean ", mean)
# an_arr = an_arr - mean
# mean = np.mean(an_arr)
# print("Updated Mean ", mean)

# # Normalizing
# norm = np.linalg.norm(an_arr)
# print("Normal ", norm)
# an_arr = an_arr / norm
# norm = np.linalg.norm(an_arr)
# print("Updated Normal ", norm)

# mean = np.mean(an_arr)
# print("Final Mean", mean)
