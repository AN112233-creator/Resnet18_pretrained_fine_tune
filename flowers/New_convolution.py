import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import scipy.io
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F 
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import torchvision.models  as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class_names = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
    "english marigold", "tiger lily", "moon orchid", "bird of paradise",
    "monkshood", "globe thistle", "snapdragon", "colt's foot", "king protea",
    "spear thistle", "yellow iris", "globe-flower", "purple coneflower",
    "peruvian lily", "balloon flower", "giant white arum lily", "fire lily",
    "pincushion flower", "fritillary", "red ginger", "grape hyacinth",
    "corn poppy", "prince of wales feathers", "stemless gentian", "artichoke",
    "sweet william", "carnation", "garden phlox", "love in the mist", "mexican aster",
    "alpine sea holly", "ruby-lipped cattleya", "cape flower", "great masterwort",
    "siam tulip", "lenten rose", "barbeton daisy", "daffodil", "sword lily",
    "poinsettia", "bolero deep blue", "wallflower", "marigold", "buttercup",
    "oxeye daisy", "common dandelion", "petunia", "wild pansy", "primula",
    "sunflower", "pelargonium", "bishop of llandaff", "gaura", "geranium",
    "orange dahlia", "pink-yellow dahlia", "cautleya spicata", "japanese anemone",
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum",
    "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania",
    "azalea", "water lily", "rose", "thorn apple", "morning glory", "passion flower",
    "lotus", "toad lily", "anthurium", "frangipani", "clematis", "hibiscus",
    "columbine", "desert-rose", "tree mallow", "magnolia", "cyclamen", "watercress",
    "canna lily", "hippeastrum", "bee balm", "ball moss", "foxglove", "bougainvillea",
    "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower",
    "trumpet creeper", "blackberry lily"
]


# Custom Dataset Class for Oxford Flowers
class FlowersDataset(Dataset):
    def __init__(self, img_dir, labels_mat, transform=None, class_names=None):
        self.img_dir = img_dir
        self.transform = transform
        self.class_names = class_names

        # Load labels from .mat file
        labels_data = scipy.io.loadmat(labels_mat)
        self.labels = labels_data['labels'][0] - 1  # MATLAB to Python index

        # List all images
        self.images = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        from PIL import Image
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        label = int(self.labels[idx])

        if self.transform:
            image = self.transform(image)

        label_name = self.class_names[label] if self.class_names else None

        return image, label, label_name  # Keep third output only if needed


# Load the setid.mat to get train/test/val indices
def load_splits(setid_mat):
    split_data = scipy.io.loadmat(setid_mat)
    train_ids = split_data['trnid'][0] - 1  # convert to 0-based indexing
    test_ids = split_data['tstid'][0] - 1
    val_ids = split_data['valid'][0] - 1
    return train_ids, test_ids, val_ids
    

# Define the image transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Path to the dataset
img_dir = './jpg'
labels_mat = './imagelabels.mat'
setid_mat = './setid.mat'



# Full dataset
full_dataset = FlowersDataset(img_dir, labels_mat, transform=train_transform, class_names=class_names)

# Load train/test/val splits
test_ids, train_ids, val_ids = load_splits(setid_mat)

# Create subsets for training and testing
train_dataset = torch.utils.data.Subset(full_dataset, train_ids)
test_dataset = torch.utils.data.Subset(full_dataset, test_ids)
val_dataset = torch.utils.data.Subset(full_dataset, val_ids)

# Load the dataset into DataLoader
training_flower_data = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,

)

test_flower_data = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False,
)

# Display the first images from the training set
samples = iter(training_flower_data)
images, labels, label_name = next(samples)
print(images.shape, labels.shape)
print(labels)
for label in labels:
    print(f"{label.item()} -> {class_names[label.item()]}")
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i].permute(1, 2, 0).numpy())
    plt.title(f" {label_name[i] }")
    plt.axis('off')
plt.show()
print(f"Number of test samples: {len(test_dataset)}")
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"number of training dataloaded per batch: {len(training_flower_data)}")
print(f"number of test dataloaded per batch: {len(test_flower_data)}")


model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  
model.fc = nn.Linear(model.fc.in_features, 102)  # Replace final layer
model = model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([
    {"params": model.fc.parameters(), "lr": 1e-3},   # new layer, train faster
    {"params": model.layer1.parameters()},
    {"params": model.layer2.parameters()},
    {"params": model.layer3.parameters()},
    {"params": model.layer4.parameters()},
], lr=1e-4)  # pretrained layers, slower

# define Scheduler
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                           factor=0.1, patience=3
                                           )

def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = training_flower_data
            else:
                model.eval()
                dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for images, labels, _ in tqdm(dataloader):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Step the scheduler on validation loss
            if phase == 'val':
                scheduler.step(epoch_loss)

                # Save best model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), "best_flowers_model.pth")

    print(f"\nBest validation accuracy: {best_acc:.4f}")
    return model
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels, _ in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Train model
model = train_model(model, criterion, optimizer, scheduler, num_epochs=10)

# Load best model
model.load_state_dict(torch.load("best_flowers_model.pth"))

# Evaluate on test set
evaluate_model(model, test_flower_data)








