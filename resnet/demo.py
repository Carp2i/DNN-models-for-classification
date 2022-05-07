import torch
from model import resnet34

model_weight_path="./resNet34.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = resnet34(num_classes=5).to(device)
net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
