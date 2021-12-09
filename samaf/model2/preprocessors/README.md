# Augmentor
- augmentor.py can be used to create augmented audio clips
- Use acoss to extract the features from generated augmented audio clips after the generation
- Example usecase
```console
python3 augmentor.py \
--meta_csv ~/Downloads/Research-Datasets/covers80/covers80_annotations.csv \
--dataset_dir ~/Downloads/Research-Datasets/covers80/covers32k/ \
--output_dataset_dir ~/Downloads/Research-Datasets/covers80/covers32k-pitch2/ \
--sample_rate 16000 \
--augmentation_type noise \
--augmentation_param 2 \
--workers 6
```