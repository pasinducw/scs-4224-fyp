import os

def listDir(path, directoriesOnly=False, filesOnly=False):
    if directoriesOnly and filesOnly:
        raise Exception(
            "Cannot have both directoriesOnly and filesOnly set to True")

    if directoriesOnly:
        return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    elif filesOnly:
        return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    else:
        return [f for f in os.listdir(path)]
