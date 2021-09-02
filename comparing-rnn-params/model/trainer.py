import copy
import math
import os

import torch
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display

from model import Model
from dataset import Covers80DatasetPerformanceChunks

import argparse

parser = argparse.ArgumentParser(
    description="RNN Parameter Trainer")
parser.add_argument("features_src", metavar="FEATURES_SRC",
                    help="path to pre-processed files")
parser.add_argument("snapshots_src", metavar="SNAPSHOTS_SRC",
                    help="path to save snapshots")
parser.add_argument("device", metavar="DEVICE", help="cuda/cpu", default="cpu")
parser.add_argument("batch_size", metavar="BATCH_SIZE",
                    help="dataset single batch size", default=512)
parser.add_argument("workers", metavar="WORKERS",
                    help="number of workers", default=4)
parser.add_argument("epochs", metavar="EPOCHS",
                    help="number of epochs to run", default=10)
parser.add_argument("state", metavar="STATE_DIM",
                    help="state dimension", default=64)


def train_model(train_dataset, validation_dataset, epochs, device, state_dimension=64, save_path="", start_state=None):

    model = Model(input_size=48, hidden_size=state_dimension).to(device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()

    history = dict(train=[], validation=[])

    best_model_weights = copy.deepcopy(model.state_dict())
    best_loss = 100000000.0

    start_epoch = 1

    if start_state:
        model.load_state_dict(start_state["model_state_dict"])
        optimizer.load_state_dict(start_state["optimizer_state_dict"])
        start_epoch = start_state["epoch"]
        history = start_state["history"]
        best_model_weights = start_state["best_model_weights"]
        best_loss = start_state["best_loss"]

    for epoch in range(start_epoch, epochs+1):
        train_losses = []
        model = model.train()
        for i, (sequence, next_frame) in enumerate(train_dataset):
            sequence = sequence.to(device)
            next_frame = next_frame.to(device)
            optimizer.zero_grad()

            next_frame_pred = model(sequence)
            loss = 1.0 * loss_fn(next_frame_pred, next_frame)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            if i % 100 == 99:
                print("Epoch {} batch {}: train loss {}".format(
                    epoch, i+1, loss.item()))

        validation_losses = []
        model = model.eval()
        with torch.no_grad():
            for i, (sequence, next_frame) in enumerate(validation_dataset):
                sequence = sequence.to(device)
                next_frame = next_frame.to(device)
                next_frame_pred = model(sequence)

                loss = 1.0 * loss_fn(next_frame_pred, next_frame)
                validation_losses.append(loss.item())
                if i % 100 == 99:
                    print("Epoch {} batch {}: validation loss {}".format(
                        epoch, i+1, loss.item()))

        train_loss = np.mean(train_losses)
        validation_loss = np.mean(validation_losses)

        history['train'].append(train_loss)
        history['validation'].append(validation_loss)

        print("Epoch {}: train loss {}, validation loss {}".format(
            epoch, train_loss, validation_loss))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "history": history,
            "best_model_weights": best_model_weights,
            "best_loss": best_loss
        }, os.path.join(save_path, "snapshot-{}.pytorch".format(epoch)))

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_model_weights = copy.deepcopy(model.state_dict())

        x = [*range(1, len(history['train'])+1)]
        plt.clf()
        plt.plot(x, history['train'], label="Train Loss")
        plt.plot(x, history['validation'], label="Validation Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title("Model Performance upto epoch {}".format(epoch))
        plt.legend()
        plt.savefig(os.path.join(
            save_path, "model-performance-{}.png".format(epoch)))

    return best_model_weights, history


def main():
    args = parser.parse_args()
    train_dataset = Covers80DatasetPerformanceChunks(
        root_dir=args.features_src, excluded_transforms=[])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=int(args.batch_size), num_workers=int(args.workers), shuffle=True)

    validation_dataset = Covers80DatasetPerformanceChunks(
        root_dir=args.features_src, excluded_transforms=[], validation=True)
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=int(args.batch_size), num_workers=int(args.workers), shuffle=False)

    device = torch.device(args.device)
    best_model, history = train_model(train_dataloader, validation_dataloader, int(args.epochs), device,
                                      int(args.state), args.snapshots_src)


if __name__ == "__main__":
    main()
