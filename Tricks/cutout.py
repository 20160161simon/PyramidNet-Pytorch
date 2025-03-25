import torch

class Cutout:
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        c, h, w = img.shape
        y = torch.randint(h, (1,))
        x = torch.randint(w, (1,))
        
        y1 = max(0, y - self.length // 2)
        y2 = min(h, y + self.length // 2)
        x1 = max(0, x - self.length // 2)
        x2 = min(w, x + self.length // 2)
        
        img[:, y1:y2, x1:x2] = 0
        return img