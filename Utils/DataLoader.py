from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder


def get_dataloader_workers():
    """Use 4 processes to read the data."""
    return 4


def load_data(train_dir, test_dir, batch_size, resize=224):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    eye_train = ImageFolder(train_dir,trans)
    eye_test = ImageFolder(test_dir, trans)
    return (data.DataLoader(eye_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(eye_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))
