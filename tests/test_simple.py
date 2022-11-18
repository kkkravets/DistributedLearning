import sys
#sys.path.insert(0, '../') # не повторять нигде


import torch
import numpy as np
import dist_framework


def train_simple_model():
    model = torch.nn.Sequential([
        torch.nn.Linear(2, 1),
        torch.nn.Sigmoid()]
    )

    criterion = torch.nn.BCELoss()

    data = {
        "feature": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "target": np.array([1, 0, 0, 0])
    }

    model_dc_framework = dist_framework.init(
        model
    )
    model_dc_framework.train(train_data = data)
    model_dc_framework.save()


if __name__ == "__main__":
    train_simple_model()
    
# Сделать валидацию
# Процесс загрузки-выгрузки модели
# 
