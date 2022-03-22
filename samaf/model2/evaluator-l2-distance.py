import os

import torch
import numpy as np
import pandas as pd
import librosa.display

from model import Model
from dataset import PerformanceChunks

import argparse
import wandb


def get_embeddings(model, dataloader, device):
    model.eval()
    db = []

    with torch.no_grad():
        for (sequence, sequence_indices, work_id, track_id) in dataloader:
            sequence = sequence.to(device)
            embeddings = model(sequence)

            # save the embeddings
            for (index, embedding) in enumerate(embeddings):
                db.append((embedding.detach().numpy(), work_id[index], track_id[index]))

    return db


def build_reference_db(model, device, config):
    reference_dataset = PerformanceChunks(
        dataset_meta_csv_path=config.reference_csv,
        base_dir=config.reference_dataset_dir,
        feature_type=config.feature_type,
        time_axis=config.time_axis,
        hop_length=config.hop_length,
        frames_per_sample=config.frames_per_sample,
        cache_limit=config.dataset_cache_limit,
    )
    reference_dataloader = torch.utils.data.DataLoader(
        reference_dataset, batch_size=config.batch_size, num_workers=config.workers, shuffle=False,
    )

    reference_db = get_embeddings(
        model, reference_dataloader, device)
    return reference_db


def query(model, device, reference_db, config):
    query_tracks = pd.read_csv(config.query_csv).values.tolist()

    all_matches = []

    # Utility function to identify a single work based on the task type
    def get_computed_work_id(work_id, track_id):
        if config.task == 'version':
            return work_id
        return "{}-{}".format(work_id, track_id)

    for index, [work_id, track_id] in enumerate(query_tracks):
        print("Processing %d/%d", index+1, len(query_tracks))
        computed_work_id = get_computed_work_id(work_id, track_id)
        query_dataset = PerformanceChunks(
            dataset_meta_csv_path=config.query_csv,
            base_dir=config.query_dataset_dir,
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

        query_embeddings = get_embeddings(
            model, query_dataloader, device,
        )

        # Find the matches
        matches = dict()

        for (query_embedding, query_work_id, query_track_id) in query_embeddings:
            best_match_index = 0
            best_distance = np.linalg.norm(query_embedding - reference_db[0][0])
            for index, (ref_embedding, ref_work_id, ref_track_id) in enumerate(reference_db):
                # Compute the distance
                distance = np.linalg.norm(query_embedding - ref_embedding)
                if distance < best_distance:
                    best_distance = distance
                    best_match_index = index
                pass
            
            matched_work_id = reference_db[best_match_index][1]
            matched_track_id = reference_db[best_match_index][2]
            matched_computed_work_id = get_computed_work_id(matched_work_id, matched_track_id)

            if matched_computed_work_id not in matches:
                    matches[matched_computed_work_id] = 0
            matches[matched_computed_work_id] += 1

        # 
        matches_list = []
        for matched_computed_work_id in matches.keys():
            matches_list.append(
                (matched_computed_work_id, matches[matched_computed_work_id]))

        dtype = [('work_id', 'S128'), ('matches', int)]
        matches_list = np.array(matches_list, dtype=dtype)
        matches_list = np.sort(matches_list, order='matches')
        # matches list contains the works that were matched, in descending order of # of votes
        matches_list = np.flip(matches_list)

        all_matches.append((computed_work_id, matches_list))

    return all_matches


def calculate_mr1(results):
    no_match_weight = 165
    ranks = []
    for (work_id, matches) in results:
        result = np.argwhere(matches['work_id'] ==
                             str.encode(work_id)).squeeze()
        has_result = result.shape != (0,)
        if has_result:
            result = result + 1
        else:
            result = no_match_weight

        ranks.append(result)

    mr1 = np.mean(ranks)
    return mr1


def calculate_accuracy(results):
    correct = 0
    incorrect = 0
    for (work_id, matches) in results:
        result = np.argwhere(matches['work_id'] ==
                             str.encode(work_id)).squeeze()
        has_result = result.shape != (0,)
        if has_result and result == 0:
            correct += 1
        else:
            incorrect += 1

    accuracy = correct/(correct+incorrect)
    return accuracy


def drive(config):
    with wandb.init(project=config.wandb_project_name, name=config.wandb_run_name,
                    job_type="evaluate", entity="pasinducw", config=config) as wandb_run:

        device = torch.device(config.device)
        model = Model(
            input_size=config.input_size, share_weights=True,
            embedding_size=config.state_dim
        ).to(device)

        model_snapshot = torch.load(
            config.model_snapshot_path, map_location=device
        )
        model.load_state_dict(model_snapshot["model"])

        if config.query_only == False:
            print("Creating reference database")
            ref_db = build_reference_db(model, device, config)
            print("Reference DB created")
            with open(os.path.join(wandb.run.dir, "db.npz"), "wb") as file:
                np.save(file, ref_db)

        print("Querying")
        query_results = query(model, device, ref_db, config)
        print("Completed Querying")
        with open(os.path.join(wandb.run.dir, "query_results.npz"), "wb") as file:
            np.save(file, query_results)

        # Get the query results
        results = {
            "accuracy": calculate_accuracy(query_results),
            "mr1": calculate_mr1(query_results),
        }
        wandb_run.log(results)
        print(results)


def main():
    parser = argparse.ArgumentParser(description="SAMAF Evaluator")

    parser.add_argument("--reference_csv", action="store", required=True,
                        help="path of reference data csv")
    parser.add_argument("--reference_dataset_dir", action="store", required=True,
                        help="root dir of reference dataset")
    parser.add_argument("--query_csv", action="store", required=True,
                        help="path of query data csv")
    parser.add_argument("--query_dataset_dir", action="store", required=True,
                        help="root dir of query dataset")
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

    parser.add_argument("--model_snapshot_path", action="store",
                        help="snapshot of the model")

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

    # If the task is audio identification, consider [work_id, track_id] as a single work
    # If the task is version identification, consider [work_id] as a single work
    parser.add_argument("--task", action="store", default="version",
                        help="audio/version")

    parser.add_argument("--query_only", action="store", default=False,
                        help="Set as True to only build the query DB")

    parser.add_argument("--snapshot_name", type=str,
                        help="Wandb run path of the snapshot")

    args = parser.parse_args()

    if args.feature_type == 'crema':
        args.input_size = 12

    print("Arguments", args)
    drive(args)


if __name__ == "__main__":
    main()
