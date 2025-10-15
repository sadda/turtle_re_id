import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from .turtle_watch_egypt import TurtleWatchEgypt_AED
from wildlife_datasets.datasets import SeaTurtleID2022 

def read_token_from_file(filepath):
    with open(filepath, 'r') as f:
        token = f.readline().strip()
    return token

def get_dataset():
    root = '/data/wildlife_datasets/data/SeaTurtleID2022'

    dataset = SeaTurtleID2022(
        root,
        category_name='flipper',
        img_load='bbox',
        load_label=True,
    )
    dataset = dataset.get_subset(~dataset.metadata['orientation'].isnull())
    dataset.metadata['label'], labels_map = pd.factorize(dataset.metadata['orientation'].values)
    dataset.col_label = 'label'

    return dataset, labels_map

def get_dataset_tw(labels_map):
    root_aed = '/data/wildlife_datasets/turtle_watch/TWE_flipperprofiles_registered ind_LR'

    dataset = TurtleWatchEgypt_AED(root_aed, load_label=True)

    labels_map2 = {'l': 'front_left', 'r': 'front_right', np.nan: np.nan}
    def extract_flipper(x):
        x_split = x.split('_')
        if len(x_split) > 0 and len(x_split[1]) == 3:
            if x_split[1][0] == 'A' and x_split[1][2] == 'F':
                return x_split[1][1].lower()
        return np.nan

    dataset.metadata['flipper'] = dataset.metadata['path'].apply(lambda x: labels_map2[extract_flipper(x)])
    dataset = dataset.get_subset(~dataset.df['flipper'].isnull())
    dataset.metadata['label'] = dataset.metadata['flipper'].apply(lambda x: np.where(x == labels_map)[0][0])
    dataset.col_label = 'label'

    return dataset

def extract_features(loader, model):
    model.eval()
    device = next(model.parameters()).device
    outputs = []
    for i, (x_batch, _) in enumerate(tqdm(loader)):
        x_batch = x_batch.to(device)
        with torch.inference_mode():
            output = model(pixel_values=x_batch)['pooler_output']
        outputs.append(output.cpu())
    return torch.cat(outputs).numpy()

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    device = next(model.parameters()).device
    total_loss = 0
    for x_batch, y_batch in tqdm(dataloader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(pixel_values=x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader):
    model.eval()
    device = next(model.parameters()).device
    preds, targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            output = model(pixel_values=x_batch)
            _, predicted = torch.max(output, 1)
            preds.extend(predicted.cpu().tolist())
            targets.extend(y_batch.tolist())
    return accuracy_score(targets, preds)

def predict(model, dataloader):
    model.eval()
    device = next(model.parameters()).device
    preds = []
    with torch.no_grad():
        for x_batch, _ in tqdm(dataloader):
            x_batch = x_batch.to(device)
            output = model(pixel_values=x_batch)
            _, predicted = torch.max(output, 1)
            preds.extend(predicted.cpu().tolist())
    return preds

class Model(torch.nn.Module):
    def __init__(self, backbone, embedding_size):
        super().__init__()

        self.backbone = backbone
        self.linear = torch.nn.Linear(embedding_size, 4)

    def forward(self, *args, **kwargs):
        x = self.backbone(*args, **kwargs)['pooler_output']
        return self.linear(x)

def get_dataloader(dataset, processor, **kwargs):
    dataset_wrapper = DatasetWrapper(dataset, processor)
    return torch.utils.data.DataLoader(dataset_wrapper, collate_fn=collate_fn, **kwargs)

def collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    if isinstance(labels[0], str):
        return pixel_values, labels
    else:
        return pixel_values, torch.tensor(labels)

class DatasetWrapper:
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        processed = self.processor(images=image, return_tensors="pt")
        processed = processed.pixel_values.squeeze(0)
        return processed, label
