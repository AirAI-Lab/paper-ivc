import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch


class RoadROIDataset(Dataset):
    def __init__(self, root, split='train', img_size=(256, 256), normalize=True):
        """
        root: 数据集根目录，例如 'RoadDataset'
        split: 'train' | 'val' | 'test'
        img_size: 输出图像大小 (H, W)
        normalize: 是否归一化到 [0,1]
        """
        self.root = root
        self.split = split
        self.img_dir = os.path.join(root, split, 'images')
        self.mask_dir = os.path.join(root, split, 'masks')

        self.img_list = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.normalize = normalize

        self.img_transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor()
        ])
        self.mask_transform = T.Compose([
            T.Resize(img_size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        mask_name = os.path.splitext(img_name)[0] + '.png'

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 读取图像
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image = self.img_transform(image)
        mask = self.mask_transform(mask)

        if self.normalize:
            image = image.float() / 255.0 if image.max() > 1 else image
            mask = (mask > 0.5).float()  # 二值化

        return image, mask


# ========== DataLoader 工具函数 ==========
def get_dataloader(root, split='train', batch_size=8, img_size=(256, 256),
                   shuffle=True, num_workers=4):
    dataset = RoadROIDataset(root=root, split=split, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle if split == 'train' else False,
                            num_workers=num_workers, pin_memory=True)
    return dataloader


if __name__ == "__main__":
    # 调试
    loader = get_dataloader("RoadDataset", split='train', batch_size=4)
    for imgs, masks in loader:
        print(imgs.shape, masks.shape)  # torch.Size([4,3,H,W]) torch.Size([4,1,H,W])
        break
