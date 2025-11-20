import os
import requests
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def read_token_from_file(filepath):
    with open(filepath, 'r') as f:
        token = f.readline().strip()
    return token

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
        for x_batch, y_batch in tqdm(dataloader):
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

def strip_suffixes(x, suffixes):
    suffixes = suffixes + [x.lower() for x in suffixes]
    while any(x.endswith(ext) for ext in suffixes):
        for ext in suffixes:
            if x.endswith(ext):
                x = x.removesuffix(ext).strip()
                break
    return x

def download_detector(
        url = 'https://huggingface.co/BVRA/TurtleDetector/resolve/main/turtle_detector.pt',
        path_model = 'models/turtle_detector.pt',
        force = False,
        ):

    if not os.path.exists(path_model) or force:
        os.makedirs(os.path.dirname(path_model), exist_ok=True)
        response = requests.get(url)
        with open(path_model, 'wb') as f:
            f.write(response.content)
