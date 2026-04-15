# ============================================
# Base import :
import io
import re
import csv
import numpy as np
import pandas as pd

# Image manipulation
import cv2 as cv
from PIL import Image

# Miscellanous
from tqdm import tqdm

import torch
import torchvision.transforms as transforms

# Directory managment
import os
import shutil
import sys
sys.path.append("..")

import config as c
# ============================================

# -------------------------
# -- Utility variables : --
# -------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dirs = {
    'train': '../datasets/processed/train',
    'val': '../datasets/processed/val',
    'test': '../datasets/processed/test'
}

transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
}

# --------------------------
# -- Homemade functions : --
# --------------------------

def tabula_rasa():
    """
    Reset all content in the dataset/processed folder
    """

    try:
        shutil.rmtree(c.PROCESSED_DATA_PATH)
        print('Former structure removed\n')
    except:
        print('First time running the project or path incorrect')


def create_project_structure():
    """
    Initiate the dataset folder architecture :
    Train, validation & test sample directory. And anomaly + normal folder in each of them
    """

    for folder in c.DATA_FOLDER:
        folder.mkdir(parents=True, exist_ok=True)

    print("Project folder structure created successfully.")


def decode_and_resize(img_bytes, size=(224, 224)):
    img = Image.open(io.BytesIO(img_bytes['bytes'])).convert("RGB")
    img = img.resize(size)
    return np.array(img)


def and_thus_df_was_born(img_dir):
    """
    Read all parquet file and turn them into dataframe
    """
    dfs = []

    for file_path in tqdm(os.listdir(img_dir), desc="Parquet read :"):
        
        full_path = os.path.join(img_dir, file_path)

        if file_path.endswith('.parquet'):
            df = pd.read_parquet(full_path)
            dfs.append(df)
    
    df1, df2, df3 = dfs[:3]

    return df1, df2, df3

    
def save_images(X, y, output_dir):
    """
    Save the given image datasets in the choosed folder
    - X : Image dataset in the form of a numpy array
    - y : Crop label stored in a numpy array
    - output_dir : pre-processed datasets folder (train, val, test)
    """

    j = 0
    directory = os.path.join(c.PROCESSED_DATA_PATH, output_dir)

    if output_dir == "train":
        image_nb = 0
    elif output_dir == "val":
        image_nb = 4795
    elif output_dir == "test":
        image_nb = 5994

    for i in tqdm(range(X.shape[0]), desc = "img saved :"):
        img = X[i]
        label = y[i]
        img_pil = Image.fromarray(img)

        if label == 1:
            img_pil.save(os.path.join(directory, f"waterbird/img_{j:05d}.png"))
        else:
            img_pil.save(os.path.join(directory, f"landbird/img_{j:05d}.png"))
        
        j += 1


def extract_img_crop(path):
    filename = os.path.basename(path)

    match = re.search(r'img_(\d+)_crop_(\d+)', filename)
    if match:
        img_id = int(match.group(1))
        crop_id = int(match.group(2))
        return (img_id, crop_id)

    return (-1, -1)


# ------------------------
# -- Github functions : --
# ------------------------

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def get_all_preds(model, loader):
    """
    Given a data sample and a model, return all prediction of the said model.
    """
    model.eval()
    with torch.no_grad():
        all_preds = torch.tensor([], device=device)
        for batch in loader:
            images = batch[0].to(device)
            preds = model(images)
            all_preds = torch.cat((all_preds, preds), dim=0)

    return all_preds


def get_confmat(targets, preds):
    stacked = torch.stack(
        (torch.as_tensor(targets, device=device),
         preds.argmax(dim=1)), dim=1
    ).tolist()
    confmat = torch.zeros(2, 2, dtype=torch.int16)
    for t, p in stacked:
        confmat[t, p] += 1

    return confmat


def get_results(confmat, classes, decimals=4):
    results = {}
    d = confmat.diagonal()

    for i, label in enumerate(classes):
        tp = d[i].item()
        tn = d.sum().item() - tp
        fp = confmat[i].sum().item() - tp
        fn = confmat[:, i].sum().item() - tp

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn) if (tp + fn) else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        f1score = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

        results[label] = {
            "accuracy": round(accuracy, decimals),
            "recall": round(recall, decimals),
            "precision": round(precision, decimals),
            "f1": round(f1score, decimals),
        }

    return results


def fit(epochs, model, criterion, optimizer, train_dl, valid_dl):
    """
    Training loop for the selectionned model
    """
    model_name = c.SELECTED_MODEL
    valid_loss_min = np.inf
    len_train, len_valid = c.LEN_TRAIN, c.LEN_VAL
    fields = [
        'epoch', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc'
    ]
    rows = []

    for epoch in range(epochs):
        train_loss, train_correct = 0, 0
        train_loop = tqdm(train_dl)

        model.train()
        for batch in train_loop:
            images, labels = batch[0].to(device), batch[1].to(device)
            preds = model(images)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += get_num_correct(preds, labels)

            train_loop.set_description(f'Epoch [{epoch+1:2d}/{epochs}]')
            train_loop.set_postfix(
                loss = loss.item(), acc=train_correct/len_train
            )
        train_loss = train_loss/len_train
        train_acc = train_correct/len_train

        model.eval()
        with torch.no_grad():
            valid_loss, valid_correct = 0, 0
            for batch in valid_dl:
                images, labels = batch[0].to(device), batch[1].to(device)
                preds = model(images)
                loss = criterion(preds, labels)
                valid_loss += loss.item() * labels.size(0)
                valid_correct += get_num_correct(preds, labels)

            valid_loss = valid_loss/len_valid
            valid_acc = valid_correct/len_valid

            rows.append([epoch, train_loss, train_acc, valid_loss, valid_acc])

            train_loop.write(
                f'\n\t\tAvg train loss: {train_loss:.6f}', end='\t'
            )
            train_loop.write(f'Avg valid loss: {valid_loss:.6f}\n')

            # save model if validation loss has decreased
            # (sometimes also referred as "Early stopping")
            if valid_loss <= valid_loss_min:
                train_loop.write('\t\tvalid_loss decreased', end=' ')
                train_loop.write(f'({valid_loss_min:.6f} -> {valid_loss:.6f})')
                train_loop.write('\t\tsaving model...\n')
                torch.save(
                    model.state_dict(),
                    f'../models/lr3e-5_{model_name}_{device}.pth'
                )
                valid_loss_min = valid_loss

    # write running results for plots
    with open(os.path.join(c.OUT_DIR, c.CSV_PATH, f'{model_name}.csv'), 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(fields)
        csv_writer.writerows(rows)


# worker init function for randomness in multiprocess dataloading
# https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
def wif(id):
    process_seed = torch.initial_seed()
    base_seed = process_seed - id
    ss = np.random.SeedSequence([id, base_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))


def load_image(path):
    image = Image.open(path)
    image = transform['eval'](image).unsqueeze(0)
    return image


def deprocess_image(image):
    image = image.cpu().numpy()
    image = np.squeeze(np.transpose(image[0], (1, 2, 0)))
    image = image * np.array((0.229, 0.224, 0.225)) + \
        np.array((0.485, 0.456, 0.406))  # un-normalize
    image = image.clip(0, 1)
    return image