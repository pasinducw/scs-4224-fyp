import copy
import math
import os
import tempfile
from multiprocessing import Pool
from functools import partial

import torch
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display

from model import Model
from dataset import PerformanceChunks

import pandas as pd

import argparse
import wandb


def calculate_accuracy(predicted, expected):
    maximumIndices = np.argmax(predicted, axis=1)
    correct = 0.0
    for (step, index) in enumerate(maximumIndices):
        if expected[step] == index:
            correct += 1.0
    return (correct / (predicted.shape[0]))


def train_model(train_dataset, validation_dataset, epochs, device, input_size, state_dimension, learning_rate, save_path, start_state=None):

    model = Model(input_size=input_size,
                  hidden_size=state_dimension).to(device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    history = dict(train=[], validation=[],
                   train_accuracy=[], validation_accuracy=[])

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

    wandb.watch(model, criterion=loss_fn, log="all")

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
            accuracy = calculate_accuracy(
                next_frame_pred.detach(), next_frame.detach()) * 100
            train_accuracies.append(accuracy)
            if i % 100 == 99:
                print("Epoch {} batch {}: train loss {}\ttrain accuracy {}%".format(
                    epoch, i+1, loss.item(), accuracy))
            wandb.log(
                {"epoch": epoch, "loss": loss.item(), "accuracy": accuracy})

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
                accuracy = calculate_accuracy(
                    next_frame_pred.detach(), next_frame.detach()) * 100
                validation_accuracies.append(accuracy)
                if i % 100 == 99:
                    print("Epoch {} batch {}: validation loss {}\tvalidation accuracy {}%".format(
                        epoch, i+1, loss.item(), accuracy))
                wandb.log(
                    {"epoch": epoch, "validation_loss": loss.item(), "validation_accuracy": accuracy})

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

        if epoch % 5 == 0:
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

    return best_model_weights, history


def work(args: map, performances: list, performance_index: int):

    work_id, track_id = performances[performance_index]

    snapshot_path = os.path.join(args['snapshots_src'], work_id, track_id)

    wandb.init(project=args['wandb_project'], entity="pasinducw", config=args)
    print("Arguments", args)
    print("(Work Id, Track Id)", (work_id, track_id))

    train_dataset = PerformanceChunks(
        # dataset_meta_csv_path=temp_dataset_meta_csv.name,
        dataset_meta_csv_path=args['meta_csv'],
        base_dir=args['dataset_dir'],
        feature_type=args['feature_type'],
        time_axis=args['time_axis'],
        hop_length=args['hop_length'],
        frames_per_sample=args['frames_per_sample'],

        work_id=work_id,
        track_id=track_id,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args['batch_size'], num_workers=args['workers'], shuffle=True)

    validation_dataset = PerformanceChunks(
        dataset_meta_csv_path=args['validation_meta_csv'],
        base_dir=args['dataset_dir'],
        feature_type=args['feature_type'],
        time_axis=args['time_axis'],
        hop_length=args['hop_length'],
        frames_per_sample=args['frames_per_sample']
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=args['batch_size'], num_workers=args['workers'], shuffle=False)

    if args['model_snapshot']:
        model_snapshot = torch.load(args['model_snapshot'])
    else:
        model_snapshot = None

    device = torch.device(args['device'])
    train_model(
        train_dataset=train_dataloader,
        validation_dataset=validation_dataloader,
        epochs=args['epochs'],
        device=device,
        input_size=args['input_size'],
        state_dimension=args['state_dim'],
        learning_rate=args['learning_rate'],
        save_path=snapshot_path,
        start_state=model_snapshot,
    )


def main():
    parser = argparse.ArgumentParser(description="RNN Parameter Trainer")

    parser.add_argument("--meta_csv", action="store", required=True,
                        help="path of metadata csv")
    parser.add_argument("--validation_meta_csv", action="store", required=True,
                        help="path of validation metadata csv")
    parser.add_argument("--dataset_dir", action="store", required=True,
                        help="root dir of dataset")
    parser.add_argument("--feature_type", action="store",
                        help="cqt/hpcp/crema", default="cqt")
    parser.add_argument("--hop_length", action="store", type=int,
                        help="hop length", default=1)
    parser.add_argument("--frames_per_sample", action="store", type=int,
                        help="frames per sample", default=100)

    parser.add_argument("--device", action="store",
                        help="cuda/cpu", default="cpu")
    parser.add_argument("--batch_size", action="store", type=int,
                        help="dataset single batch size", default=512)
    parser.add_argument("--workers", action="store", type=int,
                        help="number of workers", default=4)
    parser.add_argument("--epochs", action="store",
                        type=int, help="number of epochs to run", default=10)
    parser.add_argument("--state_dim", action="store", type=int,
                        help="state dimension", default=64)
    parser.add_argument("--learning_rate", action="store", type=float,
                        help="learning rate", default=1e-2)
    parser.add_argument('--snapshots_src', action="store", required=True,
                        help="directory to store model snapshots")

    parser.add_argument('--model_snapshot', action="store", default=None,
                        help="continue training from a previous snapshot")

    parser.add_argument('--wandb_project', action="store", default="my-test-project",
                        help="wandb project id")

    args = vars(parser.parse_args())
    args['time_axis'] = 1  # TODO: Change based on the feature type selected
    args['input_size'] = 84  # TODO: Change based on the feature type selected

    if args['feature_type'] == 'crema':
        args['input_size'] = 12

    args['hop_length'] = int(args['hop_length'])
    args['frames_per_sample'] = int(args['frames_per_sample'])
    args['batch_size'] = int(args['batch_size'])
    args['workers'] = int(args['workers'])
    args['state_dim'] = int(args['state_dim'])
    args['learning_rate'] = float(args['learning_rate'])
    args['epochs'] = int(args['epochs'])

    performances = pd.read_csv(args['meta_csv']).values.tolist()
    with Pool(int(args.workers)) as p:
        func = partial(
            work,
            args,
            performances,
        )
        p.map(func, range(len(performances)))


if __name__ == "__main__":
    main()
