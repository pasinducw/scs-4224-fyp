from threading import Thread, Lock

class ClassMapper:
    def __init__(self) -> None:
        self.mappings = dict()
        self.next_id = 1
        self.mutex = Lock()


    def get_id(self, key: any):
        if key not in self.mappings:
            try:
                self.mutex.acquire()
                if key not in self.mappings:
                    self.mappings[key] = self.next_id
                    self.next_id += 1
            finally:
                self.mutex.release()
        return self.mappings[key]
