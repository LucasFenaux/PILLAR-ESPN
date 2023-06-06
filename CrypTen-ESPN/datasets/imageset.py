from abc import abstractmethod

from torch.utils import data


class Dataset(data.Dataset):
    """ The base class for an arbitrary dataset. """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ImageSet(Dataset):
    """ The base class for an image dataset. """
    name = 'cifar10'
    num_classes = 10
    data_shape = [3, 32, 32]
    class_to_idx = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3,
                    'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

    @abstractmethod
    def get_transform(self, mode: str, **kwargs):
        # Get the transformation.
        raise NotImplementedError

    @abstractmethod
    def get_normalization(self, **kwargs):
        # Get normalization constants.
        raise NotImplementedError

    @abstractmethod
    def get_dataloader(self, mode: str, **kwargs):
        # Get a dataloader for the dataset.
        raise NotImplementedError
