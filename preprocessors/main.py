from genericpath import isdir, isfile
import AudioTransform as at
import os
from utils import listDir

MASTER = "../data/coversongs/covers32k"
TRANSFORMED = "/home/pasinducw/Downloads/Research-Datasets/covers80"

songsList = listDir(MASTER, directoriesOnly=True)
startAt = 0
progress = 0
for song in songsList:
    if startAt > progress:
        progress = progress + 1
        continue
    performances = listDir(os.path.join(MASTER, song), filesOnly=True)
    print("Performances", performances)

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
