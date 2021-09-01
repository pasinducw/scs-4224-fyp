import os
from time import time

SHOW_LOGS = False
timers = {}


def listDir(path, directoriesOnly=False, filesOnly=False):
    if directoriesOnly and filesOnly:
        raise Exception(
            "Cannot have both directoriesOnly and filesOnly set to True")
    items = []
    if directoriesOnly:
        items = [f for f in os.listdir(
            path) if os.path.isdir(os.path.join(path, f))]
    elif filesOnly:
        items = [f for f in os.listdir(
            path) if os.path.isfile(os.path.join(path, f))]
    else:
        items = [f for f in os.listdir(path)]

    filteredItems = []
    for item in items:
        if item[0] != '.':
            filteredItems.append(item)

    filteredItems.sort()
    return filteredItems


def tick(id: str):
    timers[id] = time()


def tock(id: str):
    then = timers.pop(id, 0.0)
    now = time()
    duration = (now-then) * 1000
    if duration > 1:
        if SHOW_LOGS:
            print("[{:.2f}ms \t-> {}]".format(duration, id))
