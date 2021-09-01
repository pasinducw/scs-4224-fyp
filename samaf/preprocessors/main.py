import os
from multiprocessing import Pool
from functools import partial

import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment
import numpy as np

import argparse

parser = argparse.ArgumentParser(
    description="SAMAF mp3 Songs Preprocessor")
parser.add_argument("source", metavar="SRC", help="path to mp3 files")
parser.add_argument("destination", metavar="DEST",
                    help="path to save preprocessed npy files")
parser.add_argument("workers", metavar="WORKERS",
                    help="number of workers", default=1)
parser.add_argument("start_id", metavar="START_ID",
                    help="id of first mp3 file to process", default=1)
parser.add_argument("end_id", metavar="END_ID",
                    help="id of last mp3 file to process", default=1)
parser.add_argument("sample_rate", metavar="SAMPLE_RATE",
                    help="sample rate", default=16000)
parser.add_argument("duration",
                    help="audio clip start offset", default=16000)
parser.add_argument("offset",
                    help="audio clip start offset", default=16000)


def main():
    args = parser.parse_args()
    with Pool(int(args.workers)) as p:
        func = partial(
            process_audio,
            args.source,
            args.destination,
            int(args.sample_rate),
            int(args.offset),
            int(args.duration),
        )
        p.map(func, range(int(args.start_id), int(args.end_id)))


def normalize(waveform):
    mx = np.maximum(np.amax(waveform), -np.amin(waveform))
    return waveform / (mx * 1.0)


def mfcc(waveform, plot=False):
    mfccs = librosa.feature.mfcc(
        y=waveform, sr=16000, n_mfcc=13, n_fft=400, hop_length=160)

    if plot:
        plt.figure()
        librosa.display.specshow(mfccs, x_axis="time")
        plt.colorbar()
        plt.title("MFCC")
        plt.tight_layout()
    return mfccs


def is_feature_calculated(file_id, save_directory):
    file_name = "{}.npy".format(file_id)
    if os.path.exists(os.path.join(save_directory, file_name)):
        return True
    return False


def write_features(file_id, feature_params, features, save_directory):
    file_name = "{}.npy".format(file_id)
    with open(os.path.join(save_directory, file_name), "wb") as file:
        np.save(file, (feature_params, features))


def apply_pitch_transform(waveform, sr, steps):
    return librosa.effects.pitch_shift(waveform, sr, n_steps=steps, bins_per_octave=12)


def apply_speed_transform(waveform, multiplier):
    return librosa.effects.time_stretch(waveform, multiplier)


def apply_noise(waveform):
    noise_amplitude = 0.005 * np.random.uniform() * np.amax(waveform)
    waveform = waveform.copy()
    waveform = waveform.astype(
        'float64') + noise_amplitude * np.random.normal(size=waveform.shape[0])
    return waveform


def process_audio(
        source_directory,
        save_directory,
        audio_sample_rate,
        audio_offset,
        audio_duration,
        audio_id,
):
    original = os.path.join(source_directory, "{}.mp3".format(audio_id))
    converted = "converted.{}.wav".format(audio_id)

    if os.path.exists(original):
        # if is_feature_calculated(audio_id, save_directory):
        #     print("Audio {} already processed and saved".format(audio_id))
        #     return

        print("Processing {}".format(audio_id))

        AudioSegment.from_mp3(original).export(converted, format="wav")
        waveform, sr = librosa.load(
            converted, offset=audio_offset, duration=audio_duration, sr=audio_sample_rate)
        os.remove(converted)

        audio_feature_params = []
        audio_features = []

        # original
        audio_feature_params.append("original")
        audio_features.append(mfcc(normalize(waveform)))

        # noise
        audio_feature_params.append("noised")
        audio_features.append(mfcc(normalize(apply_noise(waveform))))

        # pitch transformations
        for steps in (np.random.random(size=4) * 8 - 4):
            audio_feature_params.append("pitch:{}".format(steps))
            audio_features.append(
                mfcc(normalize(apply_pitch_transform(waveform, sr, steps))))

        # speed transforms
        for multiplier in (np.random.random(size=4) * 0.2 + 0.9):
            audio_feature_params.append("speed:{}".format(multiplier))
            audio_features.append(
                mfcc(normalize(apply_speed_transform(waveform, multiplier))))

        write_features(audio_id, audio_feature_params,
                       audio_features, save_directory)
        print("Preprocessed {}".format(audio_id))
    else:
        print("Audio file with id {} does not exist".format(audio_id))


if __name__ == "__main__":
    main()
