import torch


class SAMAF(torch.nn.Module):
    def __init__(self, n_features=13, embedding_dim=128, layers=1):
        super(SAMAF, self).__init__()
        self.n_features = n_features
        self.embedding_dim = embedding_dim

        self.rnn = torch.nn.LSTM(
            input_size=n_features, hidden_size=embedding_dim, num_layers=layers, batch_first=True)
        self.fc = torch.nn.Linear(
            in_features=self.embedding_dim, out_features=n_features)

    def forward(self, X):
        # there are this number of mfcc blocks per each audio input
        batch_size = X.shape[0]
        mfcc_blocks = X.shape[1]
        seq_length = X.shape[2]
        feature_count = self.n_features

        X = X.view(-1, 1, seq_length, feature_count).squeeze(1)
        # print("Transformed Input ", X.shape)

        # Encoder
        _, (h_n, c_n) = self.rnn(X)

        # Make a copy of the encoder hidden state and transform to [batch size, blocks, embedding size]
        embeddings = torch.clone(h_n).squeeze(0).unsqueeze(
            1).view(batch_size, mfcc_blocks, -1)
        # print("Embeddings ", embeddings.shape)

        # Decoder
        decoder_input = X[:, seq_length-1, :]
        decoder_input = decoder_input.view(-1, 1, feature_count)

        decoder_state = (h_n, c_n)
        decoder_outputs = []

        for step in range(seq_length):
            _, (h_n, c_n) = self.rnn(decoder_input, decoder_state)
            decoder_output = self.fc(h_n.squeeze(0)).unsqueeze(1)
            decoder_outputs.append(decoder_output)

            decoder_input = decoder_output.view(-1, 1, feature_count)
            decoder_state = (h_n, c_n)

        # Transform the decoder outputs to the shape of initial inputs
        decoder_outputs = torch.cat(decoder_outputs, 1)
        # print("Decoder outputs ", decoder_outputs.shape)

        decoder_outputs = decoder_outputs.unsqueeze(1)
        # print("Decoder outputs ", decoder_outputs.shape)

        decoder_outputs = decoder_outputs.view(
            batch_size, mfcc_blocks, seq_length, -1)
        # print("Decoder outputs ", decoder_outputs.shape)

        return embeddings, decoder_outputs
