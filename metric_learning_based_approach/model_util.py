import torch
import torchvision
import torch.nn as nn
import os

TRUNK_MODEL_WEIGHTS_PATH = 'saved_weights/trunk_model_weights.pth'
EMBEDDER_MODEL_WEIGHTS_PATH = 'saved_weights/embedder_model_weights.pth'


class MLP(nn.Module):
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)
    
def get_device():
    return  torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_models_dict() -> dict:
    trunk = torchvision.models.resnet101(pretrained=True)
    trunk_output_size = trunk.fc.in_features
    trunk.fc = nn.Identity()
    trunk = torch.nn.DataParallel(trunk.to(get_device()))
    embedder = torch.nn.DataParallel(MLP([trunk_output_size, 64]).to(get_device()))

    if os.path.exists(TRUNK_MODEL_WEIGHTS_PATH) and os.path.exists(EMBEDDER_MODEL_WEIGHTS_PATH):
        trunk.load_state_dict(torch.load(TRUNK_MODEL_WEIGHTS_PATH))
        embedder.load_state_dict(torch.load(EMBEDDER_MODEL_WEIGHTS_PATH))
        
    return {"trunk": trunk, "embedder": embedder}
    
    
def save_models_dict(trunk, embedder):
    torch.save(trunk.state_dict(), TRUNK_MODEL_WEIGHTS_PATH)
    torch.save(embedder.state_dict(), EMBEDDER_MODEL_WEIGHTS_PATH)