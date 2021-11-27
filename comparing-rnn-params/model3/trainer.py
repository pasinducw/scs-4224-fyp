import os

import torch

import numpy as np

from model import Model
from dataset import PerformanceChunks

import argparse
import wandb


def calculate_accuracy(predicted, expected):
    maximumIndices = np.argmax(predicted, axis=1)
    correct = 0.0
    for (step, index) in enumerate(maximumIndices):
        if expected[step] == index:
            correct += 1.0
    return (correct / (predicted.shape[0]))


def train(model, loss_fn, device, dataloader, optimizer, epoch):
    model.train()
    losses = []
    accuracies = []

    for i, (sequence, next_frame) in enumerate(dataloader):
        sequence, next_frame = sequence.to(device), next_frame.to(device)

        optimizer.zero_grad()
        next_frame_pred = model(sequence)
        loss = 1.0 * loss_fn(next_frame_pred, next_frame)
        loss.backward()
        optimizer.step()

        accuracy = calculate_accuracy(
            next_frame_pred.detach(), next_frame.detach()) * 100

        losses.append(loss.item())
        accuracies.append(accuracy)

        if i % 100 == 0:
            print("Epoch {} batch {}: train loss {}\ttrain accuracy {}%".format(
                epoch, i+1, loss.item(), accuracy))

    wandb.log({"loss": np.mean(losses),
              "accuracy": np.mean(accuracies)}, commit=False)


def validate(model, loss_fn, device, dataloader, epoch):
    model.eval()
    losses = []
    accuracies = []

    with torch.no_grad():
        for i, (sequence, next_frame) in enumerate(dataloader):
            sequence, next_frame = sequence.to(device), next_frame.to(device)
            next_frame_pred = model(sequence)

            loss = 1.0 * loss_fn(next_frame_pred, next_frame)
            losses.append(loss.item())
            accuracy = calculate_accuracy(
                next_frame_pred.detach(), next_frame.detach()) * 100
            accuracies.append(accuracy)
            if i % 100 == 0:
                print("Epoch {} batch {}: validation loss {}\tvalidation accuracy {}%".format(
                    epoch, i+1, loss.item(), accuracy))

    wandb.log(
        {"validation_loss": np.mean(losses), "validation_accuracy": np.mean(accuracy)}, commit=False)


def drive(config):
    with wandb.init(project=config.wandb_project_name, job_type="train", entity="pasinducw", config=config) as wandb_run:
        train_dataset = PerformanceChunks(
            dataset_meta_csv_path=config.meta_csv,
            base_dir=config.dataset_dir,
            feature_type=config.feature_type,
            time_axis=config.time_axis,
            hop_length=config.hop_length,
            frames_per_sample=config.frames_per_sample,
            cache_limit=config.dataset_cache_limit
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.batch_size, num_workers=config.workers, shuffle=True)

        validation_dataset = PerformanceChunks(
            dataset_meta_csv_path=config.validation_meta_csv,
            base_dir=config.dataset_dir,
            feature_type=config.feature_type,
            time_axis=config.time_axis,
            hop_length=config.hop_length,
            frames_per_sample=config.frames_per_sample,
            cache_limit=config.dataset_cache_limit
        )
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=config.batch_size, num_workers=config.workers, shuffle=False)

        device = torch.device(config.device)
        model = Model(input_size=config.input_size,
                      hidden_size=config.state_dim).to(device)

        optimizer = torch.optim.Adagrad(
            model.parameters(), lr=config.learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()

        if config.model_snapshot:
            print("Loading model snapshot")
            model_snapshot = wandb_run.use_artifact(config.model_snapshot)
            model_snapshot_dir = model_snapshot.download()
            model_snapshot = torch.load(
                os.path.join(model_snapshot_dir, "model.pth"))
            model.load_state_dict(model_snapshot["model"])

        wandb.watch(model, criterion=loss_fn, log="all")
        print("Configurations done. Commence model training")

        artifact = wandb.Artifact("{}".format(wandb_run.name), type="model")
        for epoch in range(1, config.epochs+1):
            train(model, loss_fn, device, train_dataloader, optimizer, epoch)
            validate(model, loss_fn, device, validation_dataloader, epoch)
            wandb.log({"epoch": epoch})

            if not os.path.exists(os.path.join(config.local_snapshots_dir, "checkpoints")):
                os.makedirs(os.path.join(config.local_snapshots_dir, "checkpoints"))
            model_checkpoint_path = os.path.join(
                config.local_snapshots_dir, "checkpoints", "epoch-{}.pth".format(epoch+1))

            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, model_checkpoint_path)

            artifact.add_file(model_checkpoint_path,
                              "{}/{}".format("checkpoints", "epoch-{}.pth".format(epoch+1)))

        model_path = os.path.join(config.local_snapshots_dir, "model.pth")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, model_path)
        artifact.add_file(model_path, "model.pth")
        wandb_run.log_artifact(artifact)

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

    parser.add_argument("--model_snapshot", action="store", default=None,
                        help="continue training from a previous snapshot on wandb")
    parser.add_argument("--local_snapshots_dir", action="store", required=True,
                        help="Path to store snapshots created while training, before uploading to wandb")
    parser.add_argument("--enable_checkpoints", action="store", type=bool,
                        help="Save model state at every epoch", default=True)

    parser.add_argument("--time_axis", action="store", type=int,
                        help="index of time axis", default=1)

    parser.add_argument("--input_size", action="store", type=int,
                        help="size of a single frame", default=84)

    parser.add_argument("--dataset_cache_limit", action="store", type=int,
                        help="dataset cache limit", default=100)

    parser.add_argument("--wandb_project_name", action="store", required=True,
                        help="wanDB project name")

    args = parser.parse_args()

    if args.feature_type == 'crema':
        args.input_size = 12

    print("Arguments", args)
    drive(args)


if __name__ == "__main__":
    main()
