from genericpath import isdir, isfile
import AudioTransform as at
import os
from utils import listDir

# Read the mp3 files, convert them to wav format
# Apply pitch shift and time stretch transformations
# Save into the transformed directory

MASTER = "/home/pasinducw/Downloads/Research-Datasets/covers80"
TRANSFORMED = "/home/pasinducw/Downloads/Research-Datasets/covers80_cqt"

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

        cqt = at.CQTransform().fun(sourceDirectory=sourceDirectory,
                                   targetDirectory=targetDirectory, sourceFile=performance)
        print("Extracted CQT for {}. Shape: {}".format(performance, cqt.shape))

    progress = progress + 1
    print("{}/{}".format(progress, len(songsList)))
    break
