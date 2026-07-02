import torch
import torch.nn as nn


class _Scheduler(nn.Module):
    def __init__(self, start_drop_prob, end_drop_prob, num_iters):
        super(_Scheduler, self).__init__()
        assert num_iters > 0
        assert 0 <= start_drop_prob <= 1
        assert 0 <= end_drop_prob <= 1

        self.start_drop_prob = start_drop_prob
        self.end_drop_prob = end_drop_prob
        self.num_iters = num_iters
        self.register_buffer('_current_step', torch.as_tensor(0.))

    @property
    def current_step(self):
        return self._current_step.item()

    def get_drop_prob(self):
        return NotImplementedError

    def next_drop_prob(self):
        self._current_step += 1.
        return self.get_drop_prob()


class LinearScheduler(_Scheduler):
    def __init__(self, start_drop_prob, end_drop_prob, num_iters):
        super(LinearScheduler, self).__init__(start_drop_prob, end_drop_prob, num_iters)

    def get_drop_prob(self):
        return self.current_step * (self.end_drop_prob - self.start_drop_prob) / self.num_iters


class DropBlock2d(nn.Module):
    def __init__(self, drop_prob, block_size, schedule=LinearScheduler(0, 0.9, -1)):
        super(DropBlock2d, self).__init__()
        assert block_size > 1
        self.drop_prob = drop_prob
        self.block_size = block_size

        self.max_pool = nn.MaxPool2d(block_size, 1, block_size // 2)

        self._scheduler = None
        if isinstance(schedule, _Scheduler):
            self._scheduler = schedule

    def forward(self, x):
        if not self.training or self.drop_prob <= 0:
            return x

        if self._scheduler:
            self.drop_prob = self._scheduler.next_drop_prob()

        c, h, w = x.size(1), x.size(2), x.size(3)
        gamma = self.drop_prob * (h * w) / (self.block_size ** 2) / \
                ((w - self.block_size + 1) * (h - self.block_size + 1))
        # generate mask
        mask = torch.rand(x.shape, dtype=torch.float32, device=x.device) < gamma
        mask = mask.to(x.dtype)

        mask = self.max_pool(mask)
        if self.block_size % 2 == 0:
            mask = mask[:, :, :-1, :-1]

        mask = 1 - mask

        y = x * mask * (mask.numel() / mask.sum())
        return y
