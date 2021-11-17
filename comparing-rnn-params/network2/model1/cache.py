class SimpleCache():
    cache = dict()
    keys = []
    entries_count = 0

    def __init__(self, max_entries: int = 1):
        self.max_entries = max_entries

    def get(self, key):
        if key in self.cache:
            return self.cache[key]
        return None

    def set(self, key, value):
        if self.entries_count == self.max_entries:
            del self.cache[self.keys[0]]
            del self.keys[0]
            self.entries_count -= 1

        self.cache[key] = value
        self.keys.append(key)
        self.entries_count += 1
