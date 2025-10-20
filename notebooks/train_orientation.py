import sys
sys.path.append('..')
import pickle
import torch
from transformers import AutoImageProcessor, AutoModel
from turtlewatch_egypt import get_dataset, get_dataset_tw, get_dataloader, Model
from turtlewatch_egypt import train_epoch, evaluate, read_token_from_file

n_train = 15000
n_test = 5000
n_epochs = 5
batch_size = 32
device = 'cuda:2'

token = read_token_from_file('../tokens/dinov3')
pretrained_model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(pretrained_model_name, token=token)   
backbone = AutoModel.from_pretrained(pretrained_model_name, token=token)

dataset, labels_map = get_dataset()
dataset_test_tw = get_dataset_tw(labels_map)

dataset_train = dataset.get_subset(range(n_train))
dataset_test = dataset.get_subset(range(n_train, n_train+n_test))

loader_train = get_dataloader(dataset_train, processor, batch_size=batch_size, shuffle=True)
loader_test = get_dataloader(dataset_test, processor, batch_size=batch_size, shuffle=False)
loader_test_tw = get_dataloader(dataset_test_tw, processor, batch_size=batch_size, shuffle=False)

with torch.no_grad():
    input = list(loader_train)[0][0]
    embedding_size = backbone(pixel_values=input)['pooler_output'].size(1)

model = Model(backbone, embedding_size)
model.to(device)
criterion = torch.nn.CrossEntropyLoss()

# Train the last layer for one epoch
optimizer = torch.optim.SGD(model.linear.parameters(), lr=0.001, momentum=0.9)
for param in model.backbone.parameters():
    param.requires_grad = False
model.train()
total_loss = train_epoch(model, loader_train, optimizer, criterion)

# Train the whole model
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
for param in model.backbone.parameters():
    param.requires_grad = True
for epoch in range(n_epochs):
    model.train()
    total_loss = train_epoch(model, loader_train, optimizer, criterion)
    acc1 = evaluate(model, loader_test)
    acc2 = evaluate(model, loader_test_tw)
    print(total_loss, acc1, acc2)

# Save the last checkpoint
model.to('cpu')
torch.save(model.state_dict(), 'model_weights.pth')
with open('model_data.pkl', 'wb') as file:
    pickle.dump((labels_map, embedding_size), file)