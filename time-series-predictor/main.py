import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

N = 100
L = 1000
T = 20

x = np.empty((N, L), np.float32)
x[:] = np.array(range(L)) + np.random.randint(-4*T, 4*T, N).reshape(N, 1)
y = np.sin(x/1.0/T).astype(np.float32)

# print(x.shape)
# print(y.shape)

# plt.figure(figsize=(10,8))
# plt.title("Sine wave")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.plot(np.arange(x.shape[1]), y[0,:], 'r', linewidth=2.0)
# plt.show()


class Model(nn.Module):
    def __init__(self, n_hidden=51):
        super(Model, self).__init__()
        self.n_hidden = n_hidden

        self.lstm1 = nn.LSTMCell(1, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1)

    def forward(self, X, future=0):
        outputs = []
        n_samples = X.shape[0]

        h_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        h_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)

        for input_t in torch.split(X, 1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        return outputs


def main():
    global y
    n_steps = 50
    n_hidden = 5
    lr = 0.1
    train_input = torch.from_numpy(y[3:, :-1])
    train_output = torch.from_numpy(y[3:, 1:])
    test_input = torch.from_numpy(y[:3, :-1])
    test_output = torch.from_numpy(y[:3, 1:])

    model = Model(n_hidden=n_hidden)
    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(model.parameters(), lr=lr)

    for i in range(n_steps):
        print("Step", i)

        def closure():
            optimizer.zero_grad()
            output = model(train_input)
            loss = criterion(output, train_output)
            print("Loss", loss.item())
            loss.backward()
            return loss

        optimizer.step(closure)

        # Validate
        with torch.no_grad():
            future = 1000
            pred = model(test_input, future=future)
            loss = criterion(pred[:, :-future], test_output)
            print("Test Loss", loss.item())
            y = pred.detach().numpy()

        # Plot
        plt.figure(figsize=(12, 6))
        plt.title(f"Step {i+1}")

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        n = train_input.shape[1]

        def draw(y_i, color):
            plt.plot(np.arange(n), y_i[:n], color, linewidth=2.0)
            plt.plot(np.arange(n, n+future),
                     y_i[n:], color + ":", linewidth=2.0)

        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')

        plt.savefig("predict%d.pdf" % i)
        plt.close()


if __name__ == "__main__":
    main()
