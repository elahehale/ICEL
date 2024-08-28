import os
from PIL import Image
from torch.utils.data import Dataset


class TinyImageNetValDataset(Dataset):
    def __init__(self, annotations_file, img_dir, wnid_to_label, transform=None):
        self.img_labels = []
        with open(annotations_file, 'r') as f:
            for line in f:
                img_file, wnid = line.split('\t')[:2]
                label = wnid_to_label[wnid]
                self.img_labels.append((img_file, label))

        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx][0])
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels[idx][1]
        if self.transform:
            image = self.transform(image)
        return image, label
