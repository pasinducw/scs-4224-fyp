from genericpath import isdir, isfile
import AudioTransform as at
import os
from utils import listDir

# Read the mp3 files, convert them to wav format
# Apply pitch shift and time stretch transformations
# Save into the transformed directory

DATASET = "basic"
MASTER = "/Users/pasinduwijesena/Documents/university/research/experiments/data/{}_raw".format(DATASET)
TRANSFORMED = "/Users/pasinduwijesena/Documents/university/research/experiments/data/{}".format(DATASET)

songsList = listDir(MASTER, directoriesOnly=True)
startAt = 1 # 29
endAt = 2
progress = 0
for song in songsList:
    print("Song: ", song)
    if startAt > progress:
        progress = progress + 1
        print("\tSKIP")
        continue    

    if endAt <= progress:
        print("STOP")
        break

    performances = listDir(os.path.join(MASTER, song), filesOnly=True)
    for performance in performances:
        sourceDirectory = os.path.join(MASTER, song)
        targetDirectory = os.path.join(TRANSFORMED, song)

        performanceWAV = at.WAVConversion().fun(sourceDirectory=sourceDirectory,
                                                targetDirectory=targetDirectory, sourceFile=performance)
        at.PitchShift().fun(sourceDirectory=targetDirectory,
                            targetDirectory=targetDirectory, sourceFile=performanceWAV)

        at.TimeStretch().fun(sourceDirectory=targetDirectory,
                             targetDirectory=targetDirectory, sourceFile=performanceWAV)
    progress = progress + 1
    print("{}/{}".format(progress, len(songsList)))
