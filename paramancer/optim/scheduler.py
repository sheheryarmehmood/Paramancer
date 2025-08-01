import torch


class MomentumScheduler:
    def __init__(self):
        self.restart()
    
    def __call__(self):
        t_prev = self.t
        self._update()
        return (t_prev - 1) / self.t
    
    def restart(self):
        self.t = torch.tensor(0.)
    
    def _update(self):
        self.t = (1 + torch.sqrt(1 + 4 * self.t ** 2)) / 2