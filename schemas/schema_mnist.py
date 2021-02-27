from unittest import loader
from schemas.schema_pytorch import TorchSchema

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class MnistSchema(TorchSchema):

    @staticmethod
    def list_hparams():
        return TorchSchema.list_hparams() + [
            dict(name='batch_size', type=int, default=64),
        ]

    def __init__(self, flags={}):
        super().__init__(flags)
        self.loaders = {}

    def prepare_dataset(self, set_name):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])
        is_train = set_name == 'TRAIN'
        ds = datasets.MNIST('../data',
                            train=is_train,
                            download=True,
                            transform=transform)
        bs = self.flags[f'batch_size']
        self.loaders[set_name] = DataLoader(ds, batch_size=bs)


if __name__ == "__main__":
    torch = TorchSchema({'hellp': '22'})
    mnist = MnistSchema()
    # mnist.prepare_dataset('TRAIN')
