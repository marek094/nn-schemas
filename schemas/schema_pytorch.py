from typing import DefaultDict
from unittest.loader import defaultTestLoader
from persian.schema_protocol import Schema

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torch


class TorchSchema(Schema):

    model = None

    @staticmethod
    def list_hparams():
        return Schema.list_hparams() + [
            dict(name='logdir', type=Path, default=Path() / 'runs'),
            dict(name='seed', type=int, default=42, range=(42, 52))
        ]

    def __init__(self, flags={}):
        super().__init__(flags)
        # writer
        logs = self.flags['logdir'] / self.as_hstr()
        self._writer = SummaryWriter(logs)
        self._writer_epoch = 0
        # deterministic mode
        torch.set_deterministic(True)
        torch.manual_seed(self.flags['seed'])
        # device
        self.dev = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

    def update_infoboard(self) -> None:
        for phase, metric_set in self.metrics.items():
            for name, value in metric_set.items():
                self._writer.add_scalar(f'{name}/{phase}', value,
                                        self._writer_epoch)
        self._writer_epoch += 1

    def pack_model_params(self):
        if self.model is None:
            return None
        return self.model.state_dict()

    def load_model_params(self, state_dict):
        if self.model is None:
            return None
        self.model.load_state_dict(state_dict)
