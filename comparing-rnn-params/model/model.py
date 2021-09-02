import torch


# class MultiUnitGRULayer(torch.nn.Module):
#     def __init__(self, units: int, input_size: int, hidden_size: int):
#         self.units = []
#         self.input_size = input_size
#         self.hidden_size = hidden_size

#         # Create the multiple GRU cells
#         for i in range(units):
#             self.units.append(
#                 torch.nn.GRU(input_size=input_size,
#                                  hidden_size=hidden_size)
#             )

#     def forward(self, X):
#         # X: [seq, batch, feature]

#         pass


class Model(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(Model, self).__init__()

        self.rnn = torch.nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=2,
            dropout=0.0, batch_first=True
        )
        self.fc = torch.nn.Linear(
            in_features=hidden_size, out_features=input_size,
        )

    def forward(self, X):
        # X -> [batch_size, sequence_length, feature_size]
        # rnn_output -> [batch_size, sequence_length, hidden_size]
        # prediction -> [batch_size, feature_size]
        rnn_output, h_n = self.rnn(X)
        prediction = self.fc(rnn_output[:, -1, :].squeeze(1))

        logSoftmax = torch.nn.LogSoftmax(dim=1)
        return logSoftmax(prediction)
