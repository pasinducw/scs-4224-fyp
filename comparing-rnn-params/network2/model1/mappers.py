class ClassMapper:
    def __init__(self) -> None:
        self.mappings = dict()
        self.next_id = 1

    def get_id(self, key: any):
        if key not in self.mappings:
            self.mappings[key] = self.next_id
            self.next_id += 1
        return self.mappings[key]
