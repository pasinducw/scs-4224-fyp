# What is this?

- This is the second implementation of SAMAF
- Here, we apply a semi supervised learning with triplet loss to try and improve accuracy


# Dataset Organization

- Information about the inputs are provided as `csv` files
- Feature Organization
  - [DATASET ROOT]
    - [REFERENCE]
    - [AUGMENTATION 1]
    - [AUGMENTATION 2]
    - ...
    - [AUGMENTATION n]


# Run Data Location

- Data is stored on wandb.ai
- Refer https://docs.wandb.ai/guides/track/advanced/save-restore to know how to save and restore files


# Data Evaluator

- The evaluator will generate embeddings for provided musical works

```
python3 evaluator.py  \
 --reference_csv ~/Downloads/Research-Datasets/covers80/covers80_annotations.csv \
 --query_csv ~/Downloads/Research-Datasets/covers80/covers80_annotations.csv \
 --reference_dataset_dir ~/Downloads/Research-Datasets/covers80/covers80_features/ \
 --query_dataset_dir ~/Downloads/Research-Datasets/covers80//features/speed1.05/ \
 --feature_type cqt \
 --hop_length 42 \
 --frames_per_sample 128 \
 --device cpu \
 --batch_size 512 \
 --workers 6 \
 --state_dim 128 \
 --model_snapshot_path /home/pasinducw/Documents/research/university-work-scs-4224/samaf/model2/snapshots/cross_entropy_exp1/cross_entropy_exp1-e20.pth \
 --time_axis 1 \
 --input_size 84 \
 --dataset_cache_limit 100 \
 --layers 2 \
 --wandb_project_name seq2seq-covers80-eval-1-extended-e20 \
 --wandb_run_name speed1.05 \
 --task audio
 ```