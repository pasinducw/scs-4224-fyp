from genericpath import isdir, isfile
import AudioTransform as at
import os
from utils import listDir

# Read the mp3 files, convert them to wav format
# Apply pitch shift and time stretch transformations
# Save into the transformed directory


DATASET = "covers80"
MASTER = "/Users/pasinduwijesena/Documents/university/research/experiments/data/{}".format(DATASET)
TRANSFORMED = "/Users/pasinduwijesena/Documents/university/research/experiments/data/{}_cqt".format(DATASET)

songsList = listDir(MASTER, directoriesOnly=True)
startAt = 60
endAt = 80
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

        cqt = at.CQTransform().fun(sourceDirectory=sourceDirectory,
                                   targetDirectory=targetDirectory, sourceFile=performance)
        print("Extracted CQT for {}. Shape: {}".format(performance, cqt.shape))

    progress = progress + 1
    print("{}/{}".format(progress, len(songsList)))
