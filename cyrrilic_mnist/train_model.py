import random
import shutil
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as pp
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

out_path = Path(__file__).parent / "out"
out_path.mkdir(exist_ok=True)


class DatasetSplitter:

    def __init__(
        self, zip_path="cyrillic.zip", extract_to="cyrillic", train_ratio=0.75
    ):
        self.zip_path = Path(zip_path)
        self.extract_dir = Path(extract_to)
        self.data_dir = self.extract_dir / "Cyrillic"  # исходная структура
        self.train_dir = Path("train")
        self.test_dir = Path("test")
        self.train_ratio = train_ratio

    def extract_if_needed(self):
        if not self.data_dir.exists():
            with ZipFile(self.zip_path, "r") as zip_ref:
                zip_ref.extractall(self.extract_dir)
            print(f"Архив распакован в {self.extract_dir}")
        else:
            print(f"Данные уже существуют в {self.data_dir}")

    def create_split(self):
        self.train_dir.mkdir(exist_ok=True)
        self.test_dir.mkdir(exist_ok=True)

        for class_dir in self.data_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            train_class_dir = self.train_dir / class_name
            test_class_dir = self.test_dir / class_name
            train_class_dir.mkdir(exist_ok=True)
            test_class_dir.mkdir(exist_ok=True)

            files = list(class_dir.glob("*"))
            random.shuffle(files)

            split_idx = int(len(files) * self.train_ratio)

            for file in files[:split_idx]:
                shutil.move(str(file), train_class_dir / file.name)
            for file in files[split_idx:]:
                shutil.move(str(file), test_class_dir / file.name)

            print(
                f"Класс {class_name}: {split_idx} в train, {len(files) - split_idx} в test"
            )

    def run(self):
        if (
            self.train_dir.exists()
            and self.test_dir.exists()
            and len(list(self.train_dir.iterdir())) == 34
            and len(list(self.test_dir.iterdir())) == 34
        ):
            print("Данные уже разделены на train/test")
            return

        self.extract_if_needed()
        self.create_split()
        print("Готово!")


splitter = DatasetSplitter(train_ratio=0.75)
splitter.run()


class CyrrrilicDataset(Dataset):
    def __init__(self, train=False, transforms=None):
        self.path = Path("train") if train else Path("test")

        self.files = []
        self.transforms = transforms
        self.labels = []

        classes = sorted([cls.name for cls in self.path.iterdir() if cls.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        for cls in self.path.iterdir():
            if not cls.is_dir():
                continue
            label = cls.name
            label_idx = self.class_to_idx[label]
            for img_path in cls.iterdir():
                if img_path.is_file():
                    self.files.append(img_path)
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        label = self.labels[idx]
        image = Image.open(img_path).split()[-1]
        return self.transforms(image), label


transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.RandomAffine(10, (0.1, 0.1), (0.5, 0.9), 10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

train_dataset = CyrrrilicDataset(train=True, transforms=transform)
test_dataset = CyrrrilicDataset(train=False, transforms=transform_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")


class CyrrilicCNN(nn.Module):
    def __init__(self, num_classes=34):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.selu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.selu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.selu3 = nn.LeakyReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.selu4 = nn.LeakyReLU()
        self.pool4 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.selu5 = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, num_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.selu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.selu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.selu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.selu4(x)
        x = self.pool4(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.selu5(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


model = CyrrilicCNN(num_classes=34).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

num_epochs = 20
train_loss = []
train_acc = []

model_path = out_path / "model.pth"
if not model_path.exists():
    for epoch in range(num_epochs):
        model.train()
        run_loss = 0.0
        total = 0
        correct = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()
        epoch_loss = run_loss / len(train_loader)
        epoch_acc = 100 * (correct / total)
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.3f}, Accuracy: {epoch_acc:.3f}%"
        )

    torch.save(model.cpu().state_dict(), model_path)
    model.to(device)
    print(f"Model saved to {model_path}")

    pp.figure(figsize=(12, 5))
    pp.subplot(121)
    pp.title("Training Loss")
    pp.plot(train_loss)
    pp.xlabel("Epoch")
    pp.ylabel("Loss")

    pp.subplot(122)
    pp.title("Training Accuracy")
    pp.plot(train_acc)
    pp.xlabel("Epoch")
    pp.ylabel("Accuracy (%)")
    pp.savefig(out / "train.png", dpi=300, bbox_inches="tight")
    pp.show()

else:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path} on {device}")
