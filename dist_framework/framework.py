import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict

from dist_framework import DistDataset


logger = logging.getLogger("__name__")


def init(model: torch.nn.Module, device: str, criterion: torch.nn.Module):
    return DistFramework(model, criterion)


class DistFramework:

    def __init__(self, model, criterion = torch.nn.Module, lr = 1e-3,
                 device = 'cpu', checkpoint = None):
        self.model = model
        self.optimizer = torch.optim.SGD(model.parameters(), lr = lr)
        if checkpoint:
            self.model, self.optimizer = self.load(model, optimizer, checkpoint)
        self.device = device
        self.model.to(device)

    def forward(self, feature, target):
        try:
            output = self.model(feature.to(self.device))
        except:
            logger.warning(f"feature: {feature}")
            raise
        try:
            loss = self.criterion(output, target)
        except:
            logger.warning(f"output: {output}")
            logger.warning(f"target: {target}")
            raise
        return {
            "output": output,
            "loss": loss}

    def train(self, train_data, batch_size: int = 1):
        self.model.train()
        train_data = DistDataset(train_data)
        train_dataloader = train_data.get_dataloader(batch_size = batch_size)

        for batch in train_dataloader:
            output = self.forward(*batch)
            loss = output["loss"]
            loss.backward()
            self.optimizer.step()

    def eval(self, val_data, batch_size = 1):
        self.model.eval()
        val_data = DistDataset(val_data)
        val_dataloader = val_data.get_dataloader(batch_size = batch_size)
        for batch in val_dataloader:
            with torch.no_grad():
                output = self.forward(*batch)
            loss = output["loss"]

    def test(self, test_data, batch_size = 1):
        self.model.eval()
        test_dataloader = test_data.get_dataloader(batch_size=batch_size)
        for batch in test_dataloader:
            with torch.no_grad():
                output = self.forward(*batch)
            loss = output["loss"]

    def save(self, path: Path):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, path)

    def load(self, model, optimizer, path: Path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer
