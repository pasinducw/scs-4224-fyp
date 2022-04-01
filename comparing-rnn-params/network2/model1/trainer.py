import copy
import math
import os
from faiss import knn
from parso import parse
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
from pytorch_metric_learning.utils import accuracy_calculator

class CustomAccuracyCalculator(accuracy_calculator.AccuracyCalculator):
    def calculate_precision_at_10(self, knn_labels, query_labels, **kwargs):
        if knn_labels is None:
            return 0
        return accuracy_calculator.precision_at_k(
            knn_labels, 
            query_labels[:, None], 
            10,
            self.avg_of_avgs,
            self.label_comparison_fn)

    def calculate_mr1(self, knn_labels, query_labels, **kwargs):
        ranks = 0
        for (index, labels) in enumerate(knn_labels):
            query = query_labels[index]
            rank = (labels==query).nonzero(as_tuple=True)[0][0]
            ranks += rank

        mr1 = ranks / knn_labels.shape[0] + 1
        return mr1.item()

    def requires_knn(self):
        return super().requires_knn() + ["precision_at_10", "mr1"] 





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
        anchor, positive, negative = indices_tuple
        
        max_index = 0
        if anchor.shape[0] > 0:
            max_index = np.max([np.max(anchor.numpy()), np.max(
                positive.numpy()), np.max(negative.numpy())])

        if max_index >= data.shape[0]:
            print(
                "[PREVENTED ERROR] Found index {} on the mined indices".format(max_index))
            print("Skipping the training iteration")
            return

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


def test(reference_set, query_set, model, accuracy_calculator, epoch):
    reference_embeddings, reference_labels = get_all_embeddings(reference_set, model)
    query_embeddings, query_labels = get_all_embeddings(query_set, model)
    reference_labels = reference_labels.squeeze(1)
    query_labels = query_labels.squeeze(1)

    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        query_embeddings, reference_embeddings, query_labels, reference_labels, False)
    print(
        "Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))

    wandb.log(accuracies, commit=False)


def alternative(config):
    wandb.init(project=config.wandb_project_name, name=config.wandb_run_name, entity="pasinducw", config=config)

    device = config.device
    batch_size = config.batch_size
    input_size = config.input_size
    output_size = config.output_size
    num_epochs = config.num_epochs
    learning_rate = config.learning_rate
    threshold_reducer_low = 0
    margin = 0.3
    type_of_triplets = "semihard"

    print("Initializing dataset")
    mapper = ClassMapper()
    train_dataset = PerformanceEmbeddings(dataset_meta_csv_path=config.train_meta_csv,
                                          base_dir=config.dataset_dir, class_mapper=mapper)
    query_dataset = PerformanceEmbeddings(dataset_meta_csv_path=config.query_meta_csv,
                                          base_dir=config.dataset_dir, class_mapper=mapper, mean=train_dataset.mean, norm=train_dataset.norm)
    original_dataset = PerformanceEmbeddings(dataset_meta_csv_path=config.original_meta_config,
                                             base_dir=config.dataset_dir, class_mapper=mapper, mean=train_dataset.mean, norm=train_dataset.norm)
    print("Dataset initialized")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
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
    accuracy_calculator = CustomAccuracyCalculator()
    print("accuracy_calculator initialized")

    print("Ready to run the epochs")
    wandb.watch(model, criterion=loss_func, log="all")
    for epoch in range(1, num_epochs+1):
        train(model, loss_func=loss_func, mining_func=mining_func, device=device,
              train_loader=train_loader, optimizer=optimizer, epoch=epoch)
        test(original_dataset, query_dataset,
             model, accuracy_calculator, epoch)
        wandb.log({"epoch": epoch})


def main():
    parser = argparse.ArgumentParser(description="Seq2Seq Parameter Trainer")
    parser.add_argument("--batch_size", action="store", type=int,
                        help="dataset single batch size", default=512)
    parser.add_argument("--input_size", action="store", type=int,
                        help="dataset single batch size", default=49968)
    parser.add_argument("--output_size", action="store", type=int,
                        help="dataset single batch size", default=1024)
    parser.add_argument("--num_epochs", action="store", type=int,
                        help="dataset single batch size", default=1024)
    parser.add_argument("--learning_rate", action="store", type=float,
                        help="learning rate", default=1e-2)

    parser.add_argument("--train_meta_csv", action="store", required=True,
                        help="path of train metadata csv")
    parser.add_argument("--query_meta_csv", action="store", required=True,
                        help="path of validation query metadata csv")
    parser.add_argument("--original_meta_csv", action="store", required=True,
                        help="path of all metadata csv")
    parser.add_argument("--dataset_dir", action="store", required=True,
                        help="root of the dataset")
    
    

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Arguments", args)
    alternative(args)


if __name__ == "__main__":
    main()