# ============================================
# Directory managment
import sys
sys.path.append("..")

# Custom library
import config as c

import torch
import torchvision
# ============================================

# -------------------------
# -- Utility variables : --
# -------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------
# -- List model import : --
# -------------------------

def get_vgg16(pretrained = False, out_features = None, path = None):
    model = torchvision.models.vgg16(pretrained=pretrained)
    if out_features is not None:
        model.classifier = torch.nn.Sequential(
            *list(model.classifier.children())[:-1],
            torch.nn.Linear(in_features = 4096, out_features = out_features)
        )
    if path is not None:
        model.load_state_dict(torch.load(path, map_location=device))

    return model.to(device)

def get_resnet18(pretrained = False, out_features = None, path = None):
    model = torchvision.models.resnet18(pretrained = pretrained)
    if out_features is not None:
        model.fc = torch.nn.Linear(in_features = 512, out_features = out_features)
    if path is not None:
        model.load_state_dict(torch.load(path, map_location = device))

    return model.to(device)

def get_densenet121(pretrained = False, out_features = None, path = None):
    model = torchvision.models.densenet121(pretrained = pretrained)
    if out_features is not None:
        model.classifier = torch.nn.Linear(
            in_features = 1024, out_features = out_features
        )
    if path is not None:
        model.load_state_dict(torch.load(path, map_location = device))

    return model.to(device)

def get_resnet50(pretrained = False, out_features = None, path = None):
    model = torchvision.models.resnet50(pretrained = pretrained)
    if out_features is not None:
        model.fc = torch.nn.Linear(in_features = 2048, out_features = out_features)
    if path is not None:
        model.load_state_dict(torch.load(path, map_location = device))

    return model.to(device)

# --------------------
# -- Select model : --
# --------------------

def get_selected_model(pretrained = False, out_features = None, path = None):
    """
    Get the selected model in config to be set up as the running one
    """

    if c.SELECTED_MODEL == list(c.MODEL_DIC.keys())[0]:
        model = get_vgg16(pretrained, out_features, path)

    if c.SELECTED_MODEL == list(c.MODEL_DIC.keys())[1]:
        model = get_resnet18(pretrained, out_features, path)

    if c.SELECTED_MODEL == list(c.MODEL_DIC.keys())[2]:
        model = get_densenet121(pretrained, out_features, path)

    if c.SELECTED_MODEL == list(c.MODEL_DIC.keys())[3]:
        model = get_resnet50(pretrained, out_features, path)

    return model.to(device)