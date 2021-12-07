import os

import torch
import numpy as np
import librosa.display

from model import Model
from dataset import PerformanceChunks

import argparse
import wandb


def train(model, loss_fn, device, dataloader, optimizer, epoch):
    model.train()
    losses = []

    for i, (sequence, sequence_indices, work_id, track_id) in enumerate(dataloader):
        sequence, sequence_indices = sequence.to(device), sequence_indices.to(device)

        optimizer.zero_grad()
        (embeddings, reconstructed_sequence) = model(sequence)
        loss = 1.0 * loss_fn(sequence_indices,
                             reconstructed_sequence, embeddings)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if i % 100 == 0:
            print("Epoch {} batch {}: train loss {}".format(
                epoch, i+1, loss.item()))

    wandb.log({"loss": np.mean(losses)}, commit=False)

    # Log the last available sequence and reconstructed sequence
    def get_plot(sequence):
        data = sequence.detach().cpu().numpy().transpose()
        print("Get plot shape", data.shape)
        return librosa.display.specshow(data)

    wandb.log({
        "reference_sequence": wandb.Image(get_plot(sequence[0])),
        "reconstructed_sequence": wandb.Image(get_plot(reconstructed_sequence[0]))},
        commit=False)


def validate(model, loss_fn, device, dataloader, epoch):
    losses = []

    with torch.no_grad():
        for i, (sequence, sequence_indices, work_id, track_id) in enumerate(dataloader):
            sequence, sequence_indices = sequence.to(device), sequence_indices.to(device)
            (embeddings, reconstructed_sequence) = model(sequence)

            loss = 1.0 * loss_fn(sequence_indices,
                                 reconstructed_sequence, embeddings)

            losses.append(loss.item())

            if i % 100 == 0:
                print("Epoch {} batch {}: validation loss {}".format(
                    epoch, i+1, loss.item()))

    wandb.log(
        {"validation_loss": np.mean(losses)}, commit=False)


def get_loss_function():

    autoencoder_loss = torch.nn.CrossEntropyLoss()

    def loss_fn(expected_indices_sequence, reconstructed_sequence, embeddings):
        # expected_indices_sequence -> [batch_size, sequence_length]
        # reconstructed_sequence -> [batch_size, sequence_length, feature_size]
        feature_size = reconstructed_sequence.shape[-1]

        input = reconstructed_sequence.view(-1, feature_size)
        target = expected_indices_sequence.view(-1)

        return autoencoder_loss(input, target)

    return loss_fn


def drive(config):
    with wandb.init(project=config.wandb_project_name, name=config.wandb_run_name,
                    job_type="train", entity="pasinducw", config=config) as wandb_run:
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
        model = Model(input_size=config.input_size, share_weights=not config.use_separate_models,
                      embedding_size=config.state_dim).to(device)

        optimizer = torch.optim.Adagrad(
            model.parameters(), lr=config.learning_rate)
        loss_fn = get_loss_function()

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
            if config.validate:
                validate(model, loss_fn, device, validation_dataloader, epoch)
            wandb.log({"epoch": epoch})

            model_checkpoint_dir = os.path.join(
                config.local_snapshots_dir, "checkpoints")
            if not os.path.exists(model_checkpoint_dir):
                os.makedirs(model_checkpoint_dir)
            model_checkpoint_path = os.path.join(
                model_checkpoint_dir, "model.pth")

            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, model_checkpoint_path)

            artifact.add_file(model_checkpoint_path,
                              "{}/{}".format("checkpoints", "epoch-{}.pth".format(epoch+1)))
            wandb_run.log_artifact(artifact)

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

    parser.add_argument("--time_axis", action="store", type=int,
                        help="index of time axis", default=1)

    parser.add_argument("--input_size", action="store", type=int,
                        help="size of a single frame", default=84)

    parser.add_argument("--dataset_cache_limit", action="store", type=int,
                        help="dataset cache limit", default=100)

    parser.add_argument("--layers", action="store", type=int,
                        help="number of stacked LSTM layers", default=1)

    parser.add_argument("--wandb_project_name", action="store", required=True,
                        help="wanDB project name")

    parser.add_argument("--wandb_run_name", action="store", required=False,
                        help="wanDB run name")

    parser.add_argument("--validate", action="store", type=bool, default=False,
                        help="validate after each epoch with the validation dataset")

    parser.add_argument("--use_separate_models", type=bool, default=False,
                        help="use encoder and decoder models with separate weights")

    args = parser.parse_args()

    if args.feature_type == 'crema':
        args.input_size = 12

    print("Arguments", args)
    drive(args)


if __name__ == "__main__":
    main()
