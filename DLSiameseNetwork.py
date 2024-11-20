import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
import torch
from PIL import Image
import gdown
import zipfile
import os
import requests
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import json
import csv


# Function to download a file if it doesn't exist, ignoring SSL verification
def download_if_needed(url, output_path):
    if not os.path.exists(output_path):
        print(f"Downloading {output_path}...")
        # Disable SSL verification
        response = requests.get(url, verify=False)  # Not recommended for production
        with open(output_path, 'w') as file:
            file.write(response.text)
        print(f"Downloaded {output_path}.")
    else:
        print(f"{output_path} already exists. No download needed.")

# Function to download and unzip the dataset if it doesn't exist
def download_and_unzip_if_needed(url, output_path, extract_path):
    if not os.path.exists(extract_path):  # Check if the dataset folder already exists
        if not os.path.exists(output_path):  # Check if the zip file already exists
            print("Downloading dataset...")
            gdown.download(url, output_path, quiet=False)
        else:
            print("Zip file already downloaded.")

        print("Extracting dataset...")
        print(output_path)
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
        # Iterating over each file
            for file_info in zip_ref.infolist():
                # Extract each file to the directory
                zip_ref.extract(file_info, extract_path)

        print("Extraction complete.")
    else:
        print("Dataset already exists. No download needed.")


def parse_pairs(filename):
    """ Parses a given pairs file to extract image pairs and their labels. """
    pairs = []
    with open(filename, "r") as f:
        next(f)  # Skip the header
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                # Matching pair: (person, img1, img2)
                pairs.append((f'{parts[0]}/{parts[0]}_{parts[1].zfill(4)}.jpg',
                              f'{parts[0]}/{parts[0]}_{parts[2].zfill(4)}.jpg', 1))
            elif len(parts) == 4:
                # Non-matching pair: (person1, img1, person2, img2)
                pairs.append((f'{parts[0]}/{parts[0]}_{parts[1].zfill(4)}.jpg',
                              f'{parts[2]}/{parts[2]}_{parts[3].zfill(4)}.jpg', 0))
    return pairs

class LFWPairsDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, root_dir, transform=None, augment_transform=None, augmentations=0, apply_augment=False):
        """
        Args:
            pairs: List of tuples (img_path1, img_path2, label)
            root_dir: Directory with all images.
            transform: Basic transform to be applied on all samples.
            augment_transform: Additional transformations for augmented versions.
            augmentations: Number of augmented versions to generate per image pair, dynamically.
        """
        self.pairs = pairs
        self.root_dir = root_dir
        self.transform = transform
        self.augment_transform = augment_transform
        self.augmentations = augmentations
        self.apply_augment = apply_augment  # Initialize with no augmentation

    def __len__(self):
        # If augmentations are applied, increase dataset size accordingly
        if self.apply_augment:
            return len(self.pairs) * (self.augmentations + 1)
        return len(self.pairs)

    def __getitem__(self, idx):
        actual_idx = idx % len(self.pairs)
        img_path1, img_path2, label = self.pairs[actual_idx]
        img1 = Image.open(os.path.join(self.root_dir, img_path1))
        img2 = Image.open(os.path.join(self.root_dir, img_path2))

        # Apply augmentation if enabled and the index suggests it's an augmented version
        if self.apply_augment and idx // len(self.pairs) > 0:
            if self.augment_transform:
                img1 = self.augment_transform(img1)
                img2 = self.augment_transform(img2)
        # Apply basic transformations
        elif self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (img1, img2), torch.tensor(label, dtype=torch.float32)


basic_transform = transforms.Compose([
    transforms.Resize((105, 105)),  # Resize all images to the same size
    transforms.Grayscale(),  # Convert images to grayscale
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485], std=[0.229])  # Normalize the image (single-channel statistics)
])

train_transforms = transforms.Compose([
    transforms.Resize((105, 105)),  # Resize all images to the same size
    transforms.Grayscale(),  # Convert images to grayscale
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomRotation(15),  # Randomly rotate images within +/- 15 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Random affine transformations
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Randomly change brightness and contrast
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485], std=[0.229])  # Normalize the image (grayscale statistics)
])


class SiameseNetwork(pl.LightningModule):
    def __init__(self, lr=1e-4, weight_decay=1e-2, dropout=0.3):
        super(SiameseNetwork, self).__init__()
        self.save_hyperparameters()

        self.cnn1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=64, kernel_size=10)),
            ('BatchNorm1', nn.BatchNorm2d(num_features=64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('dropout1', nn.Dropout2d(p=dropout)),  # Dropout after activation
            ('MaxPool1', nn.MaxPool2d(kernel_size=2)),
            ('conv2', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7)),
            ('BatchNorm2', nn.BatchNorm2d(num_features=128)),
            ('relu2', nn.ReLU(inplace=True)),
            ('dropout2', nn.Dropout2d(p=dropout)),  # Another dropout layer
            ('MaxPool2', nn.MaxPool2d(kernel_size=2)),
            ('conv3', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4)),
            ('BatchNorm3', nn.BatchNorm2d(num_features=128)),
            ('relu3', nn.ReLU(inplace=True)),
            ('dropout3', nn.Dropout2d(p=dropout)),  # Additional dropout
            ('MaxPool3', nn.MaxPool2d(kernel_size=2)),
            ('conv4', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4)),
            ('BatchNorm4', nn.BatchNorm2d(num_features=256)),
            ('relu4', nn.ReLU(inplace=True)),
            ('dropout4', nn.Dropout2d(p=dropout)),  # Final dropout in the conv sequence
        ]))

        # Linear layers
        self.fc1 = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout)  # Dropout before the final layer
        )
        self.out = nn.Sequential(
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )

        # Metrics
        self.train_accuracy = Accuracy(task='binary')
        self.val_accuracy = Accuracy(task='binary')
        self.test_accuracy = Accuracy(task='binary')

        self.train_loss = []
        self.train_acc = []

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.trunc_normal_(m.bias, mean=0.5, std=0.01)

            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.2)
                if m.bias is not None:
                    nn.init.trunc_normal_(m.bias, mean=0.5, std=0.01)

    def forward_one(self, x):
        x = self.cnn1(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x

    def forward(self, img1, img2):
        img1 = self.forward_one(img1)
        img2 = self.forward_one(img2)
        difference = torch.abs(img1 - img2)
        output = self.out(difference)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        # Scheduler for learning rate decay
        scheduler = {
            'scheduler': lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.99 ** epoch),
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        (img1, img2), labels = batch
        predictions = self(img1, img2)
        loss = F.binary_cross_entropy(predictions, labels.unsqueeze(1).float())
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)  # Log training loss

        preds = torch.round(predictions)
        acc = self.train_accuracy(preds, labels.unsqueeze(1))
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)  # Log training accuracy

        return loss

    def on_train_epoch_end(self, unused=None):  # Add unused=None to handle positional arguments from the trainer
        train_acc = self.train_accuracy.compute()
        train_loss = self.trainer.callback_metrics.get('train_loss', 0)  # Retrieve the average training loss

        self.train_loss.append(train_loss.item())
        self.train_acc.append(train_acc.item())

        print(f'Training Epoch End - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}')

        self.train_accuracy.reset()  # Reset metrics for the next epoch

    def validation_step(self, batch, batch_idx):
        (img1, img2), labels = batch
        predictions = self(img1, img2)
        loss = F.binary_cross_entropy(predictions, labels.unsqueeze(1).float())
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        preds = torch.round(predictions)
        acc = self.val_accuracy(preds, labels.unsqueeze(1))
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def on_validation_epoch_end(self):
        val_acc = self.val_accuracy.compute()
        val_loss = self.trainer.callback_metrics['val_loss']  # Retrieve the average validation loss

        print(f'Validation Epoch End - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')

        self.val_accuracy.reset()  # Reset metrics for the next epoch

    def test_step(self, batch, batch_idx):
        (img1, img2), labels = batch
        predictions = self(img1, img2)
        loss = F.binary_cross_entropy(predictions, labels.unsqueeze(1).float())
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)  # Log test loss

        preds = torch.round(predictions)
        acc = self.test_accuracy(preds, labels.unsqueeze(1))
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)  # Log test accuracy

        return loss

    def on_test_epoch_end(self):
        test_acc = self.test_accuracy.compute()
        test_loss = self.trainer.callback_metrics.get('test_loss', 0)  # Retrieve the average test loss

        print(f'Test Epoch End - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}')

        self.test_accuracy.reset()  # Reset metrics for the next test


# Function to convert indices to list
def indices_to_list(idx, data):
    return [data[i] for i in idx.tolist()]


if __name__ == '__main__':
    # Download the LFW dataset
    # Google Drive link and paths for dataset
    drive_url = 'https://drive.google.com/uc?id=1p1wjaqpTh_5RHfJu4vUh8JJCdKwYMHCp'
    zip_path = 'lfw_dataset.zip'
    extract_dir = 'lfw_dataset'

    # Paths for the train and test pairs files
    train_pairs_path = 'pairsDevTrain.txt'
    test_pairs_path = 'pairsDevTest.txt'
    train_pairs_url = 'https://vis-www.cs.umass.edu/lfw/pairsDevTrain.txt'
    test_pairs_url = 'https://vis-www.cs.umass.edu/lfw/pairsDevTest.txt'

    # Call the function to check and download/unzip the dataset
    download_and_unzip_if_needed(drive_url, zip_path, extract_dir)

    # Download pairs files if they don't exist, without SSL verification
    download_if_needed(train_pairs_url, train_pairs_path)
    download_if_needed(test_pairs_url, test_pairs_path)

    # Load pairs from files
    train_pairs = parse_pairs('pairsDevTrain.txt')
    test_pairs = parse_pairs('pairsDevTest.txt')

    # Root directory for the dataset
    root_dir = 'lfw_dataset/lfw2/lfw2'

    # Define hyperparameters grid
    param_grid = {
        'batch_size': [16, 32],
        'lr': [1e-3, 1e-4],
        'augmentations': [0, 4, 9],
        'dropout': [0.0, 0.3]  # Assuming you will add dropout to your model architecture
    }

    train_dataset = LFWPairsDataset(train_pairs, root_dir, transform=basic_transform)
    test_dataset = LFWPairsDataset(test_pairs, root_dir, transform=basic_transform)

    # Create all combinations of hyperparameters
    all_combinations = list(itertools.product(*(param_grid[Name] for Name in param_grid)))
    results = []

    # Define the k-fold cross-validation
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    max_epochs = 100

    # Define output directories for checkpoints and results
    os.makedirs("DL_SiameseNetwork_checkpoints", exist_ok=True)
    os.makedirs("DL_SiameseNetwork_results", exist_ok=True)

    # Initialize results storage
    results = []
    for params in [dict(zip(param_grid.keys(), values)) for values in itertools.product(*param_grid.values())]:
        print("Current Configuration:", params)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
            train_loader = DataLoader(
                LFWPairsDataset(indices_to_list(train_idx, train_pairs), root_dir,
                                transform=basic_transform, augment_transform=train_transforms,
                                augmentations=params['augmentations'], apply_augment=True),
                batch_size=params['batch_size'], shuffle=True
            )
            val_loader = DataLoader(Subset(train_dataset, val_idx.tolist()), batch_size=params['batch_size'],
                                    shuffle=False)

            model = SiameseNetwork(lr=params['lr'], dropout=params['dropout'])
            logger = TensorBoardLogger("tb_logs", name=f"model_{params}_fold_{fold}")
            checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{params}_fold_{fold}", monitor='val_acc',
                                                  mode='max', save_top_k=1)
            early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=True, mode='min')

            trainer = Trainer(
                max_epochs=max_epochs,
                logger=logger,
                callbacks=[checkpoint_callback, early_stopping],
                enable_progress_bar=False  # This will completely disable the progress bar.
            )

            start_time = time.time()
            trainer.fit(model, train_loader, val_loader)
            elapsed_time = time.time() - start_time

            val_acc = trainer.callback_metrics.get('val_acc', 0)
            fold_result = {'params': params, 'fold': fold, 'val_acc': val_acc, 'time': time.time() - start_time}
            fold_results.append(fold_result)

            val_acc = trainer.callback_metrics.get('val_acc', torch.tensor(0.0)).item()  # Convert to float
            fold_result = {
                'params': {k: v if not isinstance(v, torch.Tensor) else v.item() for k, v in params.items()},
                'fold': fold,
                'val_acc': val_acc,
                'time': elapsed_time  # assuming elapsed_time is already a float
            }
            fold_results.append(fold_result)

        avg_acc = np.mean([fr['val_acc'] for fr in fold_results])
        avg_time = np.mean([fr['time'] for fr in fold_results])
        results.append({
            'params': {k: v if not isinstance(v, torch.Tensor) else v.item() for k, v in params.items()},
            'avg_acc': avg_acc,
            'avg_time': avg_time  # Save the average time along with other metrics
        })

    csv_file_path = "DL_SiameseNetwork_results/cv_results.csv"
    json_file_path = "DL_SiameseNetwork_results/cv_results.json"
    # Save results
    with open(csv_file_path, mode='w', newline='') as file:
        fieldnames = ["params", "avg_acc", "avg_time(s)"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow({
                "params": str(result['params']),  # Convert dict to string for CSV compatibility
                "avg_acc": result['avg_acc'],
                "avg_time(s)": result['avg_time']
            })

    with open(json_file_path, 'w') as f:
        json.dump(results, f, indent=4)

    best_config = max(results, key=lambda x: x['avg_acc'])
    print("Best Configuration:", best_config['params'])

    # Retrain using the best configuration on the complete training set

    best_train_loader = DataLoader(
        LFWPairsDataset(train_pairs, root_dir, transform=basic_transform, augment_transform=train_transforms,
                        augmentations=best_config['params']['augmentations'], apply_augment=True),
        batch_size=best_config['params']['batch_size'], shuffle=True
    )

    final_epochs = 39

    final_best_model = SiameseNetwork(lr=best_config['params']['lr'], dropout=best_config['params']['dropout'])
    trainer = Trainer(max_epochs=final_epochs)
    trainer.fit(final_best_model, best_train_loader)
    trainer.test(final_best_model, DataLoader(test_dataset, batch_size=best_config['params']['batch_size']))
    test_acc = trainer.callback_metrics.get('test_acc', 0).item()


    with open('DL_SiameseNetwork_results/final_model_train_loss', 'w') as f:
        json.dump(final_best_model.train_loss, f, indent=4)

    with open('DL_SiameseNetwork_results/final_model_train_acc', 'w') as f:
        json.dump(final_best_model.train_acc, f, indent=4)

    torch.save(final_best_model.state_dict(), "DL_SiameseNetwork_results/final_model.pth")

    # --------------------------- Model Performance Analysis ----------------------------#

    # # Assuming the path to the saved model
    # model_path = "DL_SiameseNetwork_results/final_model.pth"
    #
    # # Load the final model
    # model = SiameseNetwork(lr=0.0001, dropout=0)  # Initialize the model structure first
    # model.load_state_dict(torch.load(model_path))
    # model.eval()  # Set the model to evaluation mode
    #
    # # DataLoader for the test set with a batch size of 16
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    #
    # correct = 0
    # total = 0
    # false_positives = []
    # false_negatives = []
    # true_positives = []
    # true_negatives = []
    #
    # number_of_batches = 30
    #
    # # Evaluate the model on the test data
    # # Process only the first N batches
    # for i, (images, labels) in enumerate(test_loader):
    #     if number_of_batches < i < 500:
    #         continue  # Stop after processing the specified number of batches
    #     if i > 500 + number_of_batches:
    #         break
    #     print(i)
    #     outputs = model(images[0], images[1])  # Assuming Siamese network requires pairs of images
    #     predicted = torch.round(outputs)
    #
    #     # Categorize the examples
    #     for j in range(len(labels)):
    #         img1_path = test_pairs[i][0]
    #         img2_path = test_pairs[i][1]
    #         pred = predicted[j].item()
    #         actual = labels[j].item()
    #
    #         if pred == actual:
    #             if pred == 1:
    #                 true_positives.append((img1_path, img2_path, pred, actual))
    #             else:
    #                 true_negatives.append((img1_path, img2_path, pred, actual))
    #         else:
    #             if pred == 1:
    #                 false_positives.append((img1_path, img2_path, pred, actual))
    #             else:
    #                 false_negatives.append((img1_path, img2_path, pred, actual))
    #
    #
    # # Output results for diagnostics
    # print("True Positives Count:", len(true_positives))
    # print("True Negatives Count:", len(true_negatives))
    # print("False Positives Count:", len(false_positives))
    # print("False Negatives Count:", len(false_negatives))
    #
    # # Plotting some examples from each category (TP, TN, FP, FN)
    #
    # def plot_examples(example, title):
    #     plt.figure(figsize=(8, 4))  # Adjust figure size to accommodate a single example with two images
    #     img1_path, img2_path, _, _ = example  # Get the first example
    #
    #     # Load and display the first image of the pair
    #     img1 = Image.open(root_dir + '/' + img1_path).convert('RGB')
    #     plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    #     plt.imshow(img1)
    #     plt.title(f'Image 1 - {title}')
    #     plt.axis('off')
    #
    #     # Load and display the second image of the pair
    #     img2 = Image.open(root_dir + '/' + img2_path).convert('RGB')
    #     plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    #     plt.imshow(img2)
    #     plt.title(f'Image 2 - {title}')
    #     plt.axis('off')
    #
    #     plt.suptitle(title)
    #     plt.tight_layout()
    #     plt.show()
    #
    #
    # # Example usage with the list of image pairs
    # plot_examples(true_positives[0], "True Positives")
    # plot_examples(true_negatives[2], "True Negatives")
    # plot_examples(false_positives[5], "False Positives")
    # plot_examples(false_negatives[5], "False Negatives")
