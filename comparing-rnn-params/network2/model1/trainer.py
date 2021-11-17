import copy
import math
import os
from pytorch_metric_learning.utils import accuracy_calculator

import torch
from torch import optim
from torch.utils.data import DataLoader, dataloader

import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display

from model import Model
from dataset import PerformanceEmbeddings
from mappers import ClassMapper

import argparse
import wandb

from pytorch_metric_learning import losses, miners, distances, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator


def calculate_accuracy(predicted, expected):
    maximumIndices = np.argmax(predicted, axis=1)
    correct = 0.0
    for (step, index) in enumerate(maximumIndices):
        if expected[step] == index:
            correct += 1.0
    return (correct / (predicted.shape[0]))


def plot_progress(train: list, validation: list, progress_type: str, epoch: int, save_path: str):
    x = [*range(1, len(train)+1)]
    plt.clf()
    plt.plot(x, train, label="Train {}".format(progress_type))
    plt.plot(x, validation, label="Validation {}".format(progress_type))
    plt.xlabel('Epoch')
    plt.ylabel('{}'.format(progress_type))
    plt.title("Model {} upto epoch {}".format(progress_type, epoch))
    plt.legend()
    path = os.path.join(
        save_path, "model-performance-{}-{}.png".format(epoch, progress_type))
    plt.savefig(os.path.join(path))


def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    losses = []
    for batch_index, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if batch_index % 20 == 0:
            print("Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(
                epoch, batch_index, loss, mining_func.num_triplets))

    wandb.log({"loss": np.mean(losses)}, commit=False)


def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


def test(train_set, test_set, model, accuracy_calculator, epoch):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, train_embeddings, test_labels, train_labels, False)
    print(
        "Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))

    wandb.log(accuracies, commit=False)


def alternative():
    wandb.init(project="network2", entity="pasinducw")

    device = torch.device('cpu')
    batch_size = 512
    input_size = 49968
    output_size = 1024
    num_epochs = 4096
    learning_rate = 0.1
    threshold_reducer_low = 0
    margin = 0.3
    type_of_triplets = "semihard"

    print("Initializing dataset")
    mapper = ClassMapper()
    dataset1 = PerformanceEmbeddings(dataset_meta_csv_path="/home/pasinducw/Downloads/Research-Datasets/covers80/old/embeddings/metadata.csv",
                                     base_dir="/home/pasinducw/Downloads/Research-Datasets/covers80/old/embeddings", class_mapper=mapper)
    dataset2 = PerformanceEmbeddings(dataset_meta_csv_path="/home/pasinducw/Downloads/Research-Datasets/covers80/old/embeddings/metadata.csv",
                                     base_dir="/home/pasinducw/Downloads/Research-Datasets/covers80/old/embeddings", class_mapper=mapper)
    print("Dataset initialized")
    train_loader = DataLoader(dataset1, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset2, batch_size=batch_size, shuffle=False, num_workers=4)
    print("Data loaders initialized")

    model = Model(input_size=input_size, output_size=output_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("Model initialized")

    # pytorch-learning
    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=threshold_reducer_low)
    print("distance, reducer initialized")

    loss_func = losses.TripletMarginLoss(
        margin=margin, distance=distance, reducer=reducer)
    print("loss_func initialized")
    mining_func = miners.TripletMarginMiner(
        margin=margin, distance=distance, type_of_triplets=type_of_triplets)
    print("mining func initialized)")
    accuracy_calculator = AccuracyCalculator()
    print("accuracy_calculator initialized")

    print("Ready to run the epochs")
    wandb.watch(model, criterion=loss_func, log="all")
    for epoch in range(1, num_epochs+1):
        train(model, loss_func=loss_func, mining_func=mining_func, device=device,
              train_loader=train_loader, optimizer=optimizer, epoch=epoch)
        test(dataset1, dataset2, model, accuracy_calculator, epoch)
        wandb.log({"epoch": epoch})


if __name__ == "__main__":
    alternative()
