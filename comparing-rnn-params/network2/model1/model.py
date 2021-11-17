import torch


class Model(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(Model, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_size, out_features=output_size),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=output_size, out_features=output_size),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=output_size, out_features=output_size),
            torch.nn.ReLU(),
        )

    def forward(self, X):
        return self.model(X)
