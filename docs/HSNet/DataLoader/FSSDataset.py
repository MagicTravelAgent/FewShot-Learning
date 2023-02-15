from docs.HSNet.DataLoader.CustomLoader import DatasetCustom
from torch.utils.data import DataLoader
from torchvision import transforms

class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize, length):

        cls.datasets = {
            "custom": DatasetCustom
        }

        cls.length = length

        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize

        cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(cls.img_mean, cls.img_std)])

    @classmethod
    def build_dataloader(cls, benchmark, experiment, shot=1):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility

        dataset = cls.datasets[benchmark](cls.datapath, transform=cls.transform, shot=shot, use_original_imgsize=cls.use_original_imgsize, experiment = experiment, length=cls.length)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        return dataloader