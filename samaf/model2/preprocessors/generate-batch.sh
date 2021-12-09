#!/bin/bash

meta_csv="$1"
dataset_dir="$2"
output_dir="$3"
output_prefix="$4"
final_dir="$5"

# Noise 0.01
python3 augmentor.py \
--meta_csv $meta_csv \
--dataset_dir $dataset_dir \
--output_dataset_dir "$output_dir/$output_prefix-noise0.01/" \
--sample_rate 16000 \
--augmentation_type noise \
--augmentation_param 0.01 \
--workers 6

# Noise 0.05
python3 augmentor.py \
--meta_csv $meta_csv \
--dataset_dir $dataset_dir \
--output_dataset_dir "$output_dir/$output_prefix-noise0.05/" \
--sample_rate 16000 \
--augmentation_type noise \
--augmentation_param 0.05 \
--workers 6

# Noise 0.1
python3 augmentor.py \
--meta_csv $meta_csv \
--dataset_dir $dataset_dir \
--output_dataset_dir "$output_dir/$output_prefix-noise0.1/" \
--sample_rate 16000 \
--augmentation_type noise \
--augmentation_param 0.1 \
--workers 6

# Pitch +1
python3 augmentor.py \
--meta_csv $meta_csv \
--dataset_dir $dataset_dir \
--output_dataset_dir "$output_dir/$output_prefix-pitch+1/" \
--sample_rate 16000 \
--augmentation_type pitch \
--augmentation_param 1 \
--workers 6

# Pitch +2
python3 augmentor.py \
--meta_csv $meta_csv \
--dataset_dir $dataset_dir \
--output_dataset_dir "$output_dir/$output_prefix-pitch+2/" \
--sample_rate 16000 \
--augmentation_type pitch \
--augmentation_param 2 \
--workers 6

# Pitch +3
python3 augmentor.py \
--meta_csv $meta_csv \
--dataset_dir $dataset_dir \
--output_dataset_dir "$output_dir/$output_prefix-pitch+3/" \
--sample_rate 16000 \
--augmentation_type pitch \
--augmentation_param 3 \
--workers 6

# Pitch +4
python3 augmentor.py \
--meta_csv $meta_csv \
--dataset_dir $dataset_dir \
--output_dataset_dir "$output_dir/$output_prefix-pitch+4/" \
--sample_rate 16000 \
--augmentation_type pitch \
--augmentation_param 4 \
--workers 6

# Speed 1.01
python3 augmentor.py \
--meta_csv $meta_csv \
--dataset_dir $dataset_dir \
--output_dataset_dir "$output_dir/$output_prefix-speed1.01/" \
--sample_rate 16000 \
--augmentation_type speed \
--augmentation_param 1.01 \
--workers 6

# Speed 1.05
python3 augmentor.py \
--meta_csv $meta_csv \
--dataset_dir $dataset_dir \
--output_dataset_dir "$output_dir/$output_prefix-speed1.05/" \
--sample_rate 16000 \
--augmentation_type speed \
--augmentation_param 1.05 \
--workers 6

# Speed 1.10
python3 augmentor.py \
--meta_csv $meta_csv \
--dataset_dir $dataset_dir \
--output_dataset_dir "$output_dir/$output_prefix-speed1.10/" \
--sample_rate 16000 \
--augmentation_type speed \
--augmentation_param 1.10 \
--workers 6

# Speed 0.99
python3 augmentor.py \
--meta_csv $meta_csv \
--dataset_dir $dataset_dir \
--output_dataset_dir "$output_dir/$output_prefix-speed0.99/" \
--sample_rate 16000 \
--augmentation_type speed \
--augmentation_param 0.99 \
--workers 6

# Speed 0.95
python3 augmentor.py \
--meta_csv $meta_csv \
--dataset_dir $dataset_dir \
--output_dataset_dir "$output_dir/$output_prefix-speed0.95/" \
--sample_rate 16000 \
--augmentation_type speed \
--augmentation_param 0.95 \
--workers 6

# Speed 0.90
python3 augmentor.py \
--meta_csv $meta_csv \
--dataset_dir $dataset_dir \
--output_dataset_dir "$output_dir/$output_prefix-speed0.90/" \
--sample_rate 16000 \
--augmentation_type speed \
--augmentation_param 0.90 \
--workers 6