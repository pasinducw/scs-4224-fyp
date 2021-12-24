import os

import torch
import numpy as np
import librosa.display
import time

from pytorch_metric_learning import losses, miners, distances, reducers, testers
from pytorch_metric_learning.utils import accuracy_calculator

from model import Model
from dataset import PerformanceChunks
from utils import combine_dimensions

import argparse
import wandb


def get_sequence_hash_generator_function():
    cache = dict()
    next_id = 1

    def get_sequence_id(work_ids, track_ids, offset_indexes):
        nonlocal next_id
        chunk_ids = []
        for (index, work_id) in enumerate(work_ids):
            hash_key = "%s:%s:%d" % (
                work_id, track_ids[index], offset_indexes[index])
            if hash_key not in cache:
                cache[hash_key] = next_id
                next_id += 1
            chunk_ids.append(cache[hash_key])

        return chunk_ids

    return get_sequence_id


def get_loss_function(config):
    alpha = config.loss_alpha

    threshold_reducer_low = 0  # get from config
    margin = config.triplet_loss_margin

    autoencoder_loss = torch.nn.CrossEntropyLoss()

    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=threshold_reducer_low)
    print("distance, reducer initialized")

    triplet_loss = losses.TripletMarginLoss(
        margin=margin, distance=distance, reducer=reducer)

    def loss_fn(expected_indices_sequence, reconstructed_sequence, embeddings, sequence_ids, triplet_indices):
        # expected_indices_sequence -> [batch_size * variations, sequence_length]
        # reconstructed_sequence -> [batch_size * variations, sequence_length, feature_size]
        # embeddings -> [batch_size * variations, hidden_size]
        # sequence_ids -> [batch_size * variations]

        feature_size = reconstructed_sequence.shape[-1]

        input = reconstructed_sequence.view(-1, feature_size)
        target = expected_indices_sequence.view(-1)

        ae_loss = autoencoder_loss(input, target)
        tplt_loss = triplet_loss(embeddings, sequence_ids, triplet_indices)

        # Standardize the loss values based on results from preliminary sweep (https://wandb.ai/pasinducw/seq2seq_ael_tl/sweeps/em3c9tu1)
        # Calculation: https://docs.google.com/spreadsheets/d/1KOhEaTWgfBZDI9s8hUVB1extnJ9KK50G-m4LbtV6Ucc/edit?usp=sharing
        ae_loss = (ae_loss - 3.97032104) / 0.08109961275 # Parameters calculated using training data exported from https://wandb.ai/pasinducw/seq2seq_ael_tl/sweeps/em3c9tu1
        tplt_loss = (tplt_loss - 0.06200983243) / 0.01376640325 # Parameters calculated using training data exported from https://wandb.ai/pasinducw/seq2seq_ael_tl/sweeps/em3c9tu1

        loss = alpha * ae_loss + (1-alpha) * tplt_loss
        return loss, (ae_loss, tplt_loss)

    return loss_fn


def get_mining_function(config):
    margin = config.triplet_loss_margin
    type_of_triplets = "semihard"
    distance = distances.CosineSimilarity()

    return miners.TripletMarginMiner(
        margin=margin, distance=distance, type_of_triplets=type_of_triplets)


def train(model, loss_fn, sequence_id_generator_fn, mining_fn, device, dataloader, optimizer, epoch):
    model.train()
    losses = []
    ae_losses = []
    tplt_losses = []

    times = [[], [], [], [], [], [], []]
    tasks = [
        "time_fetch_samples", "time_transform_samples", "time_send_through_network", 
        "time_compute_triplets", "time_compute_loss", "time_propagate_gradients", 
        "time_wrap_iteration"
    ]

    tick = time.time()
    for i, (sequence, sequence_indices, work_id, track_id, offset_index) in enumerate(dataloader):
        # print("Start of mini-batch %d of epoch %d" % (i, epoch))
        tock = time.time()
        times[0].append(tock-tick) # fetch samples
        # print("Elapsed %fs to fetch samples of %d" % (tock-tick, i))        

        tick = time.time()
        variations = sequence.shape[1]
        sequence = combine_dimensions(sequence, 0, 1).to(device)
        sequence_indices = combine_dimensions(
            sequence_indices, 0, 1).to(device)
        
        sequence_ids = torch.from_numpy(  # broadcast the results to all the variations
            np.array(sequence_id_generator_fn(work_id, track_id, offset_index)).reshape(-1, 1) *
            # [batch_size] * [number of variations]
            np.ones((len(offset_index), variations))
        ).view(-1).to(device)
        tock = time.time()
        times[1].append(tock-tick) # transform shapes
        # print("Elapsed %fs to transform the shapes" % (tock-tick))

        optimizer.zero_grad()

        tick = time.time()
        # Compute embeddings
        (embeddings, reconstructed_sequence) = model(sequence)
        tock = time.time()
        times[2].append(tock-tick) # send through network
        # print("Elapsed %fs to send through network" % (tock-tick))

        # Triplets
        tick = time.time()
        triplets = mining_fn(embeddings, sequence_ids)
        anchor, positive, negative = triplets

        max_index = 0
        if anchor.shape[0] > 0:
            max_index = torch.max(torch.tensor([torch.max(anchor), torch.max(
                positive), torch.max(negative)], requires_grad=False))

        if max_index >= sequence.shape[0]:
            print(
                "[PREVENTED ERROR] Found index {} on the mined indices".format(max_index))
            print("Skipping the training iteration")
            return
        tock = time.time()
        times[3].append(tock-tick) # compute triplets
        # print("Elapsed %fs to compute triplets" % (tock-tick))

        tick = time.time()
        loss, (ae_loss, tplt_loss) = loss_fn(sequence_indices,
                                             reconstructed_sequence, embeddings, sequence_ids, triplets)
        tock = time.time()
        times[4].append(tock-tick) # compute loss
        # print("Elapsed %fs to compute loss" % (tock-tick))

        tick = time.time()
        loss.backward()
        optimizer.step()
        tock = time.time()
        times[5].append(tock-tick) # propagate gradients
        # print("Elapsed %fs to propagate the gradients" % (tock-tick))
        tick = time.time()

        losses.append(loss.item())
        ae_losses.append(ae_loss.item())
        tplt_losses.append(tplt_loss.item())

        if i % 100 == 0:
            print("Epoch {} batch {}: train loss {}".format(
                epoch, i+1, loss.item()))
        
        tock = time.time()
        times[6].append(tock-tick) # wrap the process
        # print("End of iteration. Took %fs to wrap" % (tock-tick))
        tick = time.time()

    for index in range(7):
        mean_v = np.mean(times[index])
        times[index] = mean_v
        log_item = {}
        log_item[tasks[index]] = mean_v
        wandb.log(log_item, commit=False)

    wandb.log({
        "loss": np.mean(losses),
        "autoencoder_loss": np.mean(ae_losses),
        "triplet_loss": np.mean(tplt_losses)
    }, commit=False)

    # Log the last available sequence and reconstructed sequence
    def get_plot(sequence):
        data = sequence.detach().cpu().numpy().transpose()
        print("Get plot shape", data.shape)
        return librosa.display.specshow(data)

    wandb.log({
        "reference_sequence": wandb.Image(get_plot(sequence[0])),
        "reconstructed_sequence": wandb.Image(get_plot(reconstructed_sequence[0]))},
        commit=False)


def validate(model, loss_fn, sequence_id_generator_fn, mining_fn, device, dataloader, epoch):
    model.train()
    losses = []
    ae_losses = []
    tplt_losses = []

    with torch.no_grad():
        for i, (sequence, sequence_indices, work_id, track_id, offset_index) in enumerate(dataloader):
            variations = sequence.shape[1]
            sequence = combine_dimensions(sequence, 0, 1).to(device)
            sequence_indices = combine_dimensions(
                sequence_indices, 0, 1).to(device)

            sequence_ids = torch.from_numpy(  # broadcast the results to all the variations
                np.array(sequence_id_generator_fn(work_id, track_id, offset_index)).reshape(-1, 1) *
                # [batch_size] * [number of variations]
                np.ones((len(offset_index), variations))
            ).view(-1).to(device)

            # Compute embeddings
            (embeddings, reconstructed_sequence) = model(sequence)

            # Triplets
            triplets = mining_fn(embeddings, sequence_ids)
            anchor, positive, negative = triplets

            max_index = 0
            if anchor.shape[0] > 0:
                max_index = torch.max(torch.tensor([torch.max(anchor), torch.max(
                    positive), torch.max(negative)], requires_grad=False))

            if max_index >= sequence.shape[0]:
                print(
                    "[PREVENTED ERROR] Found index {} on the mined indices".format(max_index))
                print("Skipping the training iteration")
                return

            loss, (ae_loss, tplt_loss) = loss_fn(sequence_indices,
                                                 reconstructed_sequence, embeddings, sequence_ids, triplets)

            losses.append(loss.item())
            ae_losses.append(ae_loss.item())
            tplt_losses.append(tplt_loss.item())

        wandb.log({
            "validation_loss": np.mean(losses),
            "validation_autoencoder_loss": np.mean(ae_losses),
            "validation_triplet_loss": np.mean(tplt_losses)
        }, commit=False)


def drive(config):
    with wandb.init(project=config.wandb_project_name, name=config.wandb_run_name,
                    job_type="train", entity="pasinducw", config=config) as wandb_run:
        
        print("Initializing datasets and data loaders")
        train_dataset = PerformanceChunks(
            dataset_meta_csv_path=config.meta_csv,
            base_dir=config.dataset_dir,
            feature_type=config.feature_type,
            time_axis=config.time_axis,
            hop_length=config.hop_length,
            frames_per_sample=config.frames_per_sample,
            cache_limit=config.dataset_cache_limit,
            include_augmentations=True,
            augmentations_base_dir=config.augmentations_base_dir,
            augmentations=config.augmentations,
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
            cache_limit=config.dataset_cache_limit,
            include_augmentations=True,
            augmentations_base_dir=config.augmentations_base_dir,
            augmentations=config.augmentations,
        )
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=config.batch_size, num_workers=config.workers, shuffle=False)

        print("Initializing neural network and optimizers")
        device = torch.device(config.device)
        model = Model(input_size=config.input_size, share_weights=not config.use_separate_models,
                      embedding_size=config.state_dim).to(device)

        optimizer = torch.optim.Adagrad(
            model.parameters(), lr=config.learning_rate)
        chunk_id_generator_fn = get_sequence_hash_generator_function()

        loss_fn = get_loss_function(config)
        mining_fn = get_mining_function(config)

        if config.model_snapshot:
            print("Loading model snapshot")
            model_snapshot = wandb_run.restore(
                "model.pth", run_path=config.model_snapshot)
            model.load_state_dict(model_snapshot["model"])

        wandb.watch(model, criterion=loss_fn, log="all")

        print("Initialization done. Commence model training")
        for epoch in range(1, config.epochs+1):
            train(model, loss_fn, chunk_id_generator_fn, mining_fn,
                  device, train_dataloader, optimizer, epoch)
            if config.validate:
                validate(model, loss_fn, chunk_id_generator_fn, mining_fn,
                         device, validation_dataloader, epoch)
            wandb.log({"epoch": epoch})

            model_checkpoint_dir = os.path.join(
                wandb.run.dir, "checkpoints")
            if not os.path.exists(model_checkpoint_dir):
                os.makedirs(model_checkpoint_dir)
            model_checkpoint_path = os.path.join(
                model_checkpoint_dir, "checkpoint-{}.pth".format(epoch+1))

            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, model_checkpoint_path)

        model_path = os.path.join(wandb.run.dir, "model.pth")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, model_path)


def main():
    parser = argparse.ArgumentParser(description="Seq2Seq Parameter Trainer")

    parser.add_argument("--meta_csv", action="store", required=True,
                        help="path of metadata csv")
    parser.add_argument("--validation_meta_csv", action="store", required=True,
                        help="path of validation metadata csv")
    parser.add_argument("--dataset_dir", action="store", required=True,
                        help="root dir of dataset")
    parser.add_argument("--augmentations_base_dir", action="store", required=True,
                        help="base directory of augmentations")
    parser.add_argument("--augmentations", action="store", nargs="*", required=True,
                        help="augmentations list")

    parser.add_argument("--feature_type", action="store",
                        help="cqt/hpcp/crema", default="cqt")
    parser.add_argument("--hop_length", action="store", type=int,
                        help="hop length", default=1)
    parser.add_argument("--frames_per_sample", action="store", type=int,
                        help="frames per sample", default=100)

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

    # https://docs.wandb.ai/guides/track/advanced/save-restore
    parser.add_argument("--model_snapshot", action="store", default=None,
                        help="snapshot run on wandb")

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

    # Triplet loss related params
    parser.add_argument("--triplet_loss_margin", type=float, default=0.3,
                        help="triplet loss margin value")

    parser.add_argument("--loss_alpha", type=float, default=0.6,
                        help="loss multiplier")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.feature_type == 'crema':
        args.input_size = 12

    print("Arguments", args)
    drive(args)


if __name__ == "__main__":
    main()
