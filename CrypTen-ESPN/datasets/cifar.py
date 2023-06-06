from torch.utils import data
from torchvision import transforms, datasets

from .imageset import ImageSet


class CIFAR10(ImageSet):
    name = 'cifar10'
    num_classes = 10
    data_shape = [3, 32, 32]
    class_to_idx = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3,
                    'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

    def __init__(self, root="./data", auto_augment: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.train_set = datasets.CIFAR10(root=root,
                                          transform=self.get_transform(mode="train", auto_augment=auto_augment),
                                          train=True, download=True)
        self.valid_set = datasets.CIFAR10(root=root,
                                          transform=self.get_transform(mode="valid", auto_augment=auto_augment),
                                          train=False, download=True)

    def get_dataloader(self, mode: str, batch_size: int = 16, shuffle: bool = True, num_workers: int = 8, **kwargs):
        if mode == "train":
            return data.DataLoader(self.train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                   **kwargs)
        elif mode == "valid":
            return data.DataLoader(self.valid_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                   **kwargs)
        else:
            raise ValueError(f"Mode {mode} does not exist for CIFAR-10")

    def get_transform(self, mode: str, auto_augment: bool = False, **kwargs):
        if mode != 'train':
            return transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_list = [
            transforms.RandomCrop(self.data_shape[-2:], padding=self.data_shape[-1] // 8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Resize(224)
        ]
        if auto_augment:
            transform_list.insert(2, transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
        # transform_list.append(transforms.ToTensor())
        return transforms.Compose(transform_list)
