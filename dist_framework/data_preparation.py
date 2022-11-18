import torch
import pandas as pd
import numpy as np

class DistDataset:

    def __init__(self, data_dict = None, data_file = None, dtype = torch.float32, *args):
        if data_dict == None and data_file == None:
            raise ValueError("No data provided")
        if data_file:
            sep = args.get('separator', None)
            extension = data_file.split('.')[-1]
            if extension == 'txt':
                data = {}
                with open(data_file, 'r') as textfile:
                    lines = textfile.readlines()
                lines = np.array(list(map(lambda x: x.split(sep), lines)))
                data["feature"] = lines[:, 0]
                data["targets"] = lines[:, 1]
            elif extension == 'csv':
                data = pd.read_csv(data_file, sep = sep)
        else:
            data = data_dict

        self._data = {
            "feature": torch.tensor(data["feature"]).to(dtype),
            "target": torch.tensor(data["target"]).to(dtype),
        }

        if self._data["target"].dim() == 1:
            self._data["target"] = self._data["target"].unsqueeze(1)

    def get_dataloader(self, batch_size):
        dataloader = torch.utils.data.Dataset(
            self,
            batch_size=batch_size,
            collate_fn=self.default_collate_fn)
        return dataloader

    def __getitem__(self, index):
        return self._data["feature"][index], self._data["target"][index]

    def __len__(self):
        return self._data["feature"].size(0)

    @staticmethod
    def default_collate_fn(batch):
        features = []
        targets = []
        for sample in batch:
            features.append(sample[0])
            targets.append(sample[1])
        if len(features) > 1:
            return torch.concat(features, 0), torch.concat(targets, 0)
        else:
            return features[0].unsqueeze(0), targets[0].unsqueeze(0)