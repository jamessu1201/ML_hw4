import sys
import torch
from models.model import ExampleCNN
from datasets.dataloader import make_test_dataloader

import os
from tqdm import tqdm

arg=sys.argv[1:]
for i in range(len(arg)):
    if '0.' in arg[i]:
        arg[i]=float(arg[i])
    else:
        arg[i]=int(arg[i])
        
import logging        
# 設置 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(f'{arg[1:]}_output.txt'),
        logging.StreamHandler()  # 同時輸出到控制台
    ]
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

base_path = os.path.dirname(os.path.abspath(__file__))
test_data_path = os.path.join(base_path, "data", "test")
weight_path = os.path.join(base_path, "weights", "weight.pth")

# load model and use weights we saved before
model = ExampleCNN()
model.load_state_dict(torch.load(weight_path, weights_only=True))
model = model.to(device)

# make dataloader for test data
test_loader = make_test_dataloader(test_data_path,arg[2])

predict_correct = 0
model.eval()
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        output = model(data)
        predict_correct += (output.data.max(1)[1] == target.data).sum()
        
    accuracy = 100. * predict_correct / len(test_loader.dataset)
logging.info(f'Test accuracy: {accuracy:.4f}%')