import numpy as np
class Engine:
    def __init__(self, config_path=None): self.load_s = 0.0
    def default_anchor(self):
        return np.zeros((1,480,640,3), np.uint8), np.zeros(6, np.float32)
    def generate(self, **kw):
        return [np.zeros((480,640,3), np.uint8) for _ in range(16)]
