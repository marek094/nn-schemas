from schemas.schema_cifar10 import Cifar10Schema

import torch as T
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR


class CnnCifar10Schema(Cifar10Schema):

    @staticmethod
    def make_cnn(c=64, num_classes=10):
        ''' Returns a 5-layer CNN with width parameter c. '''

        class Flatten(nn.Module):

            def forward(self, x):
                return x.view(x.size(0), x.size(1))

        return nn.Sequential(
            # Layer 0
            nn.Conv2d(3, c, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(c),
            nn.ReLU(),

            # Layer 1
            nn.Conv2d(c, c * 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 2
            nn.Conv2d(c * 2,
                      c * 4,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 3
            nn.Conv2d(c * 4,
                      c * 8,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 4
            nn.MaxPool2d(4),
            Flatten(),
            nn.Linear(c * 8, num_classes, bias=True))

    @staticmethod
    def list_hparams():
        return Cifar10Schema.list_hparams() + [
            dict(name='epochs', type=int, default=80),
            dict(name='lr', type=float, default=0.1, range=(0.01, 0.21, 0.01)),
            dict(name='weight_decay', type=float, default=5e-4),
            dict(name='lr_decay', type=int, default=1000),
            dict(name='width', type=int, default=10, range=(2, 80, 4)),
        ]

    def __init__(self, flags={}) -> None:
        super().__init__(flags)

    def prepare_model(self):
        self.model = self.make_cnn(self.flags['width'], num_classes=10)
        self.model = self.model.to(self.dev)

    def prepare_criterium(self):
        self.optim = T.optim.Adam(self.model.parameters(),
                                  lr=self.flags['lr'],
                                  weight_decay=self.flags['weight_decay'])

        self.scheduler = StepLR(self.optim,
                                step_size=self.flags['lr_decay'],
                                gamma=0.1)

        self.crit = nn.CrossEntropyLoss()

    def epoch_range(self):
        return range(self.flags['epochs'])

    def run_batches(self, set_name):
        if set_name == 'TRAIN':
            self.metrics[set_name] = self._run_batches_train(set_name)
        else:
            self.metrics[set_name] = self._run_batches_valid(set_name)

    def _run_batches_train(self, set_name):
        self.model.train()
        train_loss, correct, total = 0, 0, 0
        for inputs, targets in self.loaders[set_name]:
            inputs, targets = inputs.to(self.dev), targets.to(self.dev)
            self.optim.zero_grad()
            outputs = self.model(inputs)
            loss = self.crit(outputs, targets)
            loss.backward()
            self.optim.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return dict(loss=train_loss / total, acc=100. * correct / total)

    def _run_batches_valid(self, set_name):
        self.model.eval()
        test_loss, correct, total = 0, 0, 0
        with T.no_grad():
            for inputs, targets in self.loaders[set_name]:
                inputs, targets = inputs.to(self.dev), targets.to(self.dev)
                outputs = self.model(inputs)
                loss = self.crit(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return dict(loss=test_loss / total, acc=100. * correct / total)
