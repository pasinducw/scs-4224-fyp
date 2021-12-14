import os
from multiprocessing import Pool
from functools import partial

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import essentia.standard as estd
import argparse


def apply_pitch_transform(waveform, sr, steps):
    steps = float(steps)
    return librosa.effects.pitch_shift(waveform, sr, n_steps=steps, bins_per_octave=24)


def apply_speed_transform(waveform, sr, multiplier):
    multiplier = float(multiplier)
    return librosa.effects.time_stretch(waveform, multiplier)


def apply_noise(waveform, sr, percentage=0.005):
    print(waveform.shape)
    percentage = float(percentage)
    noise_amplitude = percentage * np.random.uniform() * np.amax(waveform)
    waveform = waveform.copy()
    waveform = waveform.astype(
        'float64') + noise_amplitude * np.random.normal(size=waveform.shape[0])
    return waveform


def process_audio(
        config,
        dataset,
        row,
):
    transformations = {
        "pitch": apply_pitch_transform,
        "speed": apply_speed_transform,
        "noise": apply_noise,
    }

    (work_id, track_id) = dataset[row]
    original = os.path.join(config.dataset_dir, work_id,
                            "{}.mp3".format(track_id))

    if os.path.exists(original):
        print("Processing {}={}".format(work_id, track_id))

        converted_dir = os.path.join(config.output_dataset_dir, work_id)
        converted_path = os.path.join(converted_dir, "{}.mp3".format(track_id))

        if not os.path.exists(converted_dir):
            os.makedirs(converted_dir)

        waveform = estd.MonoLoader(
            filename=original, sampleRate=config.sample_rate)()
        

        transformed_waveform = transformations[config.augmentation_type](
            waveform, config.sample_rate, config.augmentation_param)

        estd.MonoWriter(filename=converted_path,
                         sampleRate=config.sample_rate, format="mp3")(transformed_waveform)


def drive(config):
    source_tracks = pd.read_csv(config.meta_csv, dtype=str).values.tolist()
    # for index in range(len(source_tracks)):
    #   process_audio(config, source_tracks, index)

    with Pool(int(config.workers)) as p:
        func = partial(
            process_audio,
            config,
            source_tracks,
        )

        p.map(func, range(len(source_tracks)))


def main():
    parser = argparse.ArgumentParser(description="RNN Parameter Trainer")

    parser.add_argument("--meta_csv", action="store", required=True,
                        help="path of metadata csv")
    parser.add_argument("--dataset_dir", action="store", required=True,
                        help="root dir of dataset")
    parser.add_argument("--output_dataset_dir", action="store", required=True,
                        help="output root dir of dataset")
    
    parser.add_argument("--sample_rate", action="store", required=True, type=int,
                        help="sample rate of audio files")
    parser.add_argument("--augmentation_type", action="store",
                        help="pitch/speed/noise", default="noise")
    parser.add_argument("--augmentation_param", action="store",
                        help="corresponding parameter for the augmentation", default="0.005")
    parser.add_argument("--workers", action="store", type=int,
                        help="number of workers", default=4)

    args = parser.parse_args()
    print("Arguments", args)
    drive(args)


if __name__ == "__main__":
    main()
