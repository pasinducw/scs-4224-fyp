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


def calculate_accuracy(predicted, expected):
    maximumIndices = np.argmax(predicted, axis=1)
    correct = 0.0
    for (step, index) in enumerate(maximumIndices):
        if expected[step] == index:
            correct += 1.0
    return (correct / (predicted.shape[0]))


def plot_progress(train:list, validation:list, progress_type:str, epoch:int, save_path:str):
    x = [*range(1, len(train)+1)]
    plt.clf()
    plt.plot(x, train, label="Train {}".format(progress_type))
    plt.plot(x, validation, label="Validation {}".format(progress_type))
    plt.xlabel('Epoch')
    plt.ylabel('{}'.format(progress_type))
    plt.title("Model {} upto epoch {}".format(progress_type, epoch))
    plt.legend()
    path = os.path.join(save_path, "model-performance-{}-{}.png".format(epoch, progress_type))
    plt.savefig(os.path.join(path))

def train_model(train_dataset, validation_dataset, epochs, device, state_dimension=64, save_path="", start_state=None):

    model = Model(input_size=48, hidden_size=state_dimension).to(device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()

    history = dict(train=[], validation=[], train_accuracy=[], validation_accuracy=[])

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
        train_accuracies = []
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
            accuracy = calculate_accuracy(next_frame_pred.detach(), next_frame.detach()) * 100
            train_accuracies.append(accuracy)
            if i % 100 == 99:
                print("Epoch {} batch {}: train loss {}\ttrain accuracy {}%".format(
                    epoch, i+1, loss.item(), accuracy))

        validation_losses = []
        validation_accuracies = []
        model = model.eval()
        with torch.no_grad():
            for i, (sequence, next_frame) in enumerate(validation_dataset):
                sequence = sequence.to(device)
                next_frame = next_frame.to(device)
                next_frame_pred = model(sequence)

                loss = 1.0 * loss_fn(next_frame_pred, next_frame)
                validation_losses.append(loss.item())
                accuracy = calculate_accuracy(next_frame_pred.detach(), next_frame.detach()) * 100
                validation_accuracies.append(accuracy)
                if i % 100 == 99:
                    print("Epoch {} batch {}: validation loss {}\tvalidation accuracy {}%".format(
                        epoch, i+1, loss.item(), accuracy))

        train_loss = np.mean(train_losses)
        validation_loss = np.mean(validation_losses)
        train_accuracy = np.mean(train_accuracies)
        validation_accuracy = np.mean(validation_accuracies)

        history['train'].append(train_loss)
        history['validation'].append(validation_loss)
        history['train_accuracy'].append(train_accuracy)
        history['validation_accuracy'].append(validation_accuracy)
        

        print("Epoch {}: train loss {}, validation loss {}, train accuracy {}%, validation accuracy {}%".format(
            epoch, train_loss, validation_loss, train_accuracy, validation_accuracy))
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

        plot_progress(train=history['train'], validation=history['validation'], progress_type='Loss', epoch=epoch, save_path=save_path)
        plot_progress(train=history['train_accuracy'], validation=history['validation_accuracy'], progress_type='Accuracy', epoch=epoch, save_path=save_path)

    return best_model_weights, history


def main():
    args = parser.parse_args()
    
    excluded_transforms = [
        "_PITCH_SHIFT_0", "_PITCH_SHIFT_1", "_PITCH_SHIFT_2", "_PITCH_SHIFT_3", "_PITCH_SHIFT_4",
        "_PITCH_SHIFT_5", "_PITCH_SHIFT_6", "_PITCH_SHIFT_7",
        "_TIME_STRETCH_0", "_TIME_STRETCH_1", "_TIME_STRETCH_2", "_TIME_STRETCH_3", "_TIME_STRETCH_4",
        "_TIME_STRETCH_5", "_TIME_STRETCH_6", "_TIME_STRETCH_7",  "_TIME_STRETCH_8",
    ]

    train_dataset = Covers80DatasetPerformanceChunks(
        root_dir=args.features_src, excluded_transforms=excluded_transforms)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=int(args.batch_size), num_workers=int(args.workers), shuffle=True)

    # Include extra augmentations
    excluded_transforms = [
        "_PITCH_SHIFT_0", "_PITCH_SHIFT_1", "_PITCH_SHIFT_2", "_PITCH_SHIFT_3", "_PITCH_SHIFT_4",
        "_TIME_STRETCH_0", "_TIME_STRETCH_1", "_TIME_STRETCH_2", "_TIME_STRETCH_3", "_TIME_STRETCH_4",
    ]

    validation_dataset = Covers80DatasetPerformanceChunks(
        root_dir=args.features_src, excluded_transforms=excluded_transforms, validation=True)
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=int(args.batch_size), num_workers=int(args.workers), shuffle=False)

    device = torch.device(args.device)
    best_model, history = train_model(train_dataloader, validation_dataloader, int(args.epochs), device,
                                      int(args.state), args.snapshots_src)


if __name__ == "__main__":
    main()
