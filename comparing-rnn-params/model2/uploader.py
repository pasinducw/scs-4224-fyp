import wandb
import argparse
import torch
import os


def drive(config):
    raw_model_snapshot = torch.load(
        os.path.join(config.snapshot_path))

    torch.save({
        "model": raw_model_snapshot['model_state_dict'],
        "optimizer": raw_model_snapshot['optimizer_state_dict'],
    }, "{}.v3".format(config.snapshot_path))

    with wandb.init(project=config.wandb_project_name, job_type="train", entity="pasinducw", config=config) as wandb_run:
        artifact = wandb.Artifact("{}".format(wandb_run.name), type="model")
        artifact.add_file("{}.v3".format(config.snapshot_path), "model.pth")
        wandb_run.log_artifact(artifact)


def main():
    parser = argparse.ArgumentParser(
        description="Wandb Model Artifact Uploader")

    parser.add_argument("--snapshot_path", action="store", required=True,
                        help="path of model snapshot")

    parser.add_argument("--wandb_project_name", action="store", required=True,
                        help="wanDB project name")

    args = parser.parse_args()

    print("Arguments", args)
    drive(args)


if __name__ == "__main__":
    main()
