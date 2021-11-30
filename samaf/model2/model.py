import torch
from dataset import PerformanceChunks


class Model(torch.nn.Module):
    def __init__(self, input_size=13, embedding_size=128, layers=1, share_weights=True):
        super(Model, self).__init__()
        self.n_features = input_size
        self.embedding_size = embedding_size

        self.projection = torch.nn.Linear(
            in_features=self.embedding_size, out_features=input_size)

        self.encoder = torch.nn.LSTM(
            input_size=input_size, hidden_size=embedding_size, num_layers=layers, batch_first=True,
        )

        if share_weights != True:
            self.decoder = torch.nn.LSTM(
                input_size=input_size, hidden_size=embedding_size, num_layers=layers, batch_first=True,
            )

        self.share_weights = share_weights

    def forward(self, X):
        # X -> [batch_size, sequence_length, feature_size]

        # there are this number of mfcc blocks per each audio input
        batch_size, sequence_length, feature_size = X.shape[0], X.shape[1], X.shape[2]

        encoder = self.encoder
        if self.share_weights:
            decoder = self.encoder
        else:
            decoder = self.decoder

        # Encode
        _, (h_n, c_n) = encoder(X)

        # Make a copy of the encoder hidden state and transform to [batch size, blocks, embedding size]
        embeddings = torch.clone(h_n[-1])  # TODO: Why clone?
        print("Embeddings ", embeddings.shape)

        # return embeddings
        # Decode
        decoder_input = torch.clone(X[:, 0, :])

        decoder_state = (h_n, c_n)
        decoder_outputs = []

        for step in range(sequence_length):
            # output -> [batch_size, sequence_length(1 here), embedding_size]
            # h_n -> [number_of_layers, batch_size, embedding_size]
            # c_n -> [number_of_layers, batch_size, embedding_size]
            output, (h_n, c_n) = decoder(decoder_input, decoder_state)
            decoder_output = self.fc(output.squeeze(1))
            decoder_outputs.append(decoder_output)

            decoder_input = decoder_output.unsqueeze(1)
            decoder_state = (h_n, c_n)

        # Transform the decoder outputs to the shape of initial inputs
        decoder_outputs = torch.cat(decoder_outputs, 1)
        print("Decoder outputs ", decoder_outputs.shape)

        decoder_outputs = decoder_outputs.unsqueeze(1)
        print("Decoder outputs ", decoder_outputs.shape)

        return embeddings, decoder_outputs


def test(config):
    dataset = PerformanceChunks(
        dataset_meta_csv_path=config.meta_csv,
        base_dir=config.dataset_dir,
        feature_type=config.feature_type,
        time_axis=config.time_axis,
        hop_length=config.hop_length,
        frames_per_sample=config.frames_per_sample,
        cache_limit=config.dataset_cache_limit
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, num_workers=config.workers, shuffle=True)

    device = torch.device(config.device)
    model = Model(input_size=config.input_size,
                  hidden_size=config.hidden_size).to(device)

    model.train()

    for i, (sequence, next_frame) in enumerate(dataloader):
        sequence, next_frame = sequence.to(device), next_frame.to(device)
        model(sequence)


if __name__ == "__main__":
    config = {
        "meta_csv": "",
        "dataset_dir": "",
        "feature_type": "cqt",
        "time_axis": 1,
        "hop_length": 42,
        "frames_per_sample": 100,
        "cache_limit": 80,
        "workers": 1,
        "device": "cpu",
        "input_size": 84,
        "hidden_size": 128
    }

    test(config)
