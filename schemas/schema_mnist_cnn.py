from schemas.schema_mnist import MnistSchema

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


class CnnMnistSchema(MnistSchema):

    class Net(nn.Module):

        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output

    @staticmethod
    def list_hparams():
        return MnistSchema.list_hparams() + [
            dict(name='epochs', type=int, default=14),
            dict(name='lr', type=float, default=1.0),
            dict(name='gamma', type=float, default=0.99),
        ]

    def prepare_model(self):
        self.model = CnnMnistSchema.Net().to(self.dev)

    def prepare_criterium(self):
        self.optim = optim.Adadelta(self.model.parameters(),
                                    lr=self.flags['lr'])
        self.scheduler = StepLR(self.optim,
                                step_size=1,
                                gamma=self.flags['gamma'])

    def epoch_range(self):
        return range(self.flags['epochs'])

    def run_batches(self, set_name):
        if set_name == 'TRAIN':
            loss, acc = self._run_batches_train(set_name)
        else:
            loss, acc = self._run_batches_valid(set_name)

        print(loss, acc)
        self.metrics[set_name] = dict(loss=loss, accuracy=acc)

    def _run_batches_train(self, set_name):
        self.model.train()
        loss, correct = 0, 0
        for data, target in self.loaders[set_name]:
            data, target = data.to(self.dev), target.to(self.dev)
            self.optim.zero_grad()
            output = self.model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss_g = F.nll_loss(output, target)
            loss_g.backward()
            loss += loss_g.item()
            self.optim.step()
            self.scheduler.step()
            # break
        n = float(len(self.loaders[set_name].dataset))
        return loss / n, correct / n

    def _run_batches_valid(self, set_name):
        self.model.eval()
        loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.loaders[set_name]:
                data, target = data.to(self.dev), target.to(self.dev)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                loss += F.nll_loss(output, target, reduction='sum').item()
                # break
        n = float(len(self.loaders[set_name].dataset))
        return loss / n, correct / n
