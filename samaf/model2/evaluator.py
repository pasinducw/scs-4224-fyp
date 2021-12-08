import os

import torch
import numpy as np
import pandas as pd
import librosa.display

from model import Model
from dataset import PerformanceChunks

import argparse
import wandb


def get_hashes_dict(model, dataloader, device, hash_fn):
    model.eval()
    db = dict()

    with torch.no_grad():
        for (sequence, sequence_indices, work_id, track_id) in dataloader:
            sequence = sequence.to(device)
            embeddings = model(sequence)

            # convert the embeddings to hashes
            hashes = hash_fn(embeddings.detach().numpy())

            # save the hashes
            for (index, hash) in enumerate(hashes):
                if hash not in db:
                    db[hash] = []
                db[hash].append((work_id[index], track_id[index], hash))

    return db


def build_hash_fn(pivot=0.0):
    def threshold(value):
        if value > pivot:
            return True
        return False
    vectorized_threshold = np.vectorize(threshold)

    def hash_fn(embeddings):
        # embeddings -> [batch_size, hidden_size]
        batch_size, hidden_size = embeddings.shape

        boolean_values = vectorized_threshold(embeddings).astype(bool)
        hashes = np.zeros(batch_size)

        for row in range(batch_size):
            hash = 0.0
            for (index, value) in enumerate(boolean_values[row]):
                if value == True:
                    hash += 1 << index
            hashes[row] = hash

        return hashes
    return hash_fn


def build_reference_db(model, device, config):
    reference_dataset = PerformanceChunks(
        dataset_meta_csv_path=config.reference_csv,
        base_dir=config.dataset_dir,
        feature_type=config.feature_type,
        time_axis=config.time_axis,
        hop_length=config.hop_length,
        frames_per_sample=config.frames_per_sample,
        cache_limit=config.dataset_cache_limit,
    )
    reference_dataloader = torch.utils.data.DataLoader(
        reference_dataset, batch_size=config.batch_size, num_workers=config.workers, shuffle=False,
    )

    reference_db = get_hashes_dict(
        model, reference_dataloader, device, build_hash_fn(0.0))
    return reference_db


def query(model, device, reference_db, config):
    query_tracks = pd.read_csv(config.query_csv).values.tolist()

    for [work_id, track_id] in query_tracks:
        query_dataset = PerformanceChunks(
            dataset_meta_csv_path=config.query_csv,
            base_dir=config.dataset_dir,
            feature_type=config.feature_type,
            time_axis=config.time_axis,
            hop_length=2,  # config.hop_length,
            frames_per_sample=config.frames_per_sample,
            cache_limit=config.dataset_cache_limit,
            work_id=work_id,
            track_id=track_id,
        )

        query_dataloader = torch.utils.data.DataLoader(
            query_dataset, batch_size=config.batch_size, num_workers=config.workers, shuffle=False,
        )

        query_hashes = get_hashes_dict(
            model, query_dataloader, device, build_hash_fn(0.0),
        ).keys()

        # Find the matches
        matches = dict()

        no_match_count = 0
        match_count = 0

        for hash in query_hashes:
            matched_entries = []
            if hash in reference_db:
                matched_entries = reference_db[hash]
                match_count += 1
            else:
                no_match_count += 1

            for (matched_work_id, matched_track_id, matched_hash) in matched_entries:
                if matched_work_id not in matches:
                    matches[matched_work_id] = 0
                matches[matched_work_id] += 1
        matches_list = []
        for matched_work_id in matches.keys():
            matches_list.append((matched_work_id, matches[matched_work_id]))

        dtype = [('work_id', 'S128'), ('matches', int)]
        matches_list = np.array(matches_list, dtype=dtype)
        matches_list = np.sort(matches_list, order='matches')
        # matches list contains the works that were matched, in descending order of # of votes
        matches_list = np.flip(matches_list)


def drive(config):
    with wandb.init(project=config.wandb_project_name, name=config.wandb_run_name,
                    job_type="evaluate", entity="pasinducw", config=config) as wandb_run:
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
    parser = argparse.ArgumentParser(description="SAMAF Evaluator")

    parser.add_argument("--reference_csv", action="store", required=True,
                        help="path of reference data csv")
    parser.add_argument("--query_csv", action="store", required=True,
                        help="path of query data csv")
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
    parser.add_argument("--state_dim", action="store", type=int,
                        help="state dimension", default=64)

    parser.add_argument("--model_snapshot", action="store", default=None,
                        help="snapshot of the model from wandb")

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

    parser.add_argument("--use_separate_models", type=bool, default=False,
                        help="use encoder and decoder models with separate weights")

    args = parser.parse_args()

    if args.feature_type == 'crema':
        args.input_size = 12

    print("Arguments", args)
    drive(args)


if __name__ == "__main__":
    main()
