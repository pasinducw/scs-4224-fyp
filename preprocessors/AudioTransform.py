import librosa as la
import soundfile as sf
from pydub import AudioSegment
import os

STANDARD_SAMPLE_RATE = 22050


class WAVConversion:
    def __init__(self, sr=STANDARD_SAMPLE_RATE):
        self.sr = sr

    def fun(self, sourceDirectory: str, targetDirectory: str, sourceFile: str):
        fileName = '.'.join(sourceFile.split('.')[:-1])
        format = sourceFile.split('.')[-1]

        # Create the target directory if it doesn't exist
        if not os.path.exists(targetDirectory):
            os.makedirs(targetDirectory)

        # Throw if not a mp3
        if format != 'mp3':
            raise "Unexpected format {}. Expected a mp3".format(format)

        convertedFile = "{}_WAV.wav".format(fileName)

        convertedFilePath = os.path.join(targetDirectory, convertedFile)
        originalFilePath = os.path.join(sourceDirectory, sourceFile)

        AudioSegment.from_mp3(originalFilePath).export(convertedFilePath)
        return convertedFile


class PitchShift:

    def __init__(self, nPitch=4, sr=STANDARD_SAMPLE_RATE, binPerOctave=24):
        self.nPitch = nPitch
        self.sr = sr
        self.binPerOctave = binPerOctave

    def fun(self, sourceDirectory: str, targetDirectory: str, sourceFile: str):
        fileName = '.'.join(sourceFile.split('.')[:-1])

        audio, sr2 = la.load(os.path.join(
            sourceDirectory, sourceFile), sr=self.sr)
        changes = list(range(-self.nPitch, 0)) + list(range(1, self.nPitch+1))
        specs = [la.effects.pitch_shift(audio,
                                        sr=self.sr, n_steps=i, bins_per_octave=self.binPerOctave)
                 for i in changes]
        print("[PITCH_SHIFT] generated additional {} samples".format(len(specs)))

        names = list()
        for i in range(len(specs)):
            name = "{}_PITCH_SHIFT_{}.wav".format(fileName, i)
            sf.write(os.path.join(targetDirectory, name), specs[i], self.sr)
            names.append(name)
        return names


class TimeStretch:

    def __init__(self, maxStretch=2, interval=0.2,
                 sr=STANDARD_SAMPLE_RATE, binPerOctave=24):

        self.maxStretch = maxStretch
        self.interval = interval
        self.sr = sr
        self.binPerOctave = binPerOctave

    def fun(self, sourceDirectory: str, targetDirectory: str, sourceFile: str):
        fileName = '.'.join(sourceFile.split('.')[:-1])

        audio, sr2 = la.load(os.path.join(
            sourceDirectory, sourceFile), sr=self.sr)
        changes = [float(i) * self.interval for i in
                   range(1, int(self.maxStretch/self.interval))]
        specs = [la.effects.time_stretch(audio, rate=i) for i in changes]

        print("[TIME_STRETCH] generated additional {} samples".format(len(specs)))

        names = list()
        for i in range(len(specs)):
            name = "{}_TIME_STRETCH_{}.wav".format(fileName, i)
            sf.write(os.path.join(targetDirectory, name), specs[i], self.sr)
            names.append(name)
        return names
