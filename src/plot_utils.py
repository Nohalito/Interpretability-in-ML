# ============================================
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Directory managment
import os
import sys
sys.path.append("..")

# Custom library
import config as c
from utils import deprocess_image
# ============================================

def plot_summary(path):
    """
    Use the model csv that was tracking its progression through the epochs to plot the train and valid loss.
    """
    data = pd.read_csv(path)
    filename = path.split('/')[-1][:-4]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 4))

    ax1.plot(data['epoch'], data['train_loss'], label = 'Train Loss')
    ax1.plot(data['epoch'], data['valid_loss'], label = 'Valid Loss')
    ax1.axhline(data['valid_loss'].min(),
                linestyle = (0, (5, 10)), linewidth=0.5)
    ax1.axvline(data['valid_loss'].idxmin(),
                linestyle = (0, (5, 10)), linewidth=0.5)

    ax2.plot(data['epoch'], data['train_acc'], label = 'Train Accuracy')
    ax2.plot(data['epoch'], data['valid_acc'], label = 'Valid Accuracy')

    ax1.legend()
    ax1.set_title('Running Loss', fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True)

    ax2.legend()
    ax2.set_title('Running Accuracy', fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)

    plt.tight_layout()
    fig.savefig(os.path.join(c.OUT_DIR, c.SUMMARY_PATH, f'{filename}.png'))
    plt.show()
    plt.close()


def plot_confmat(train_mat, test_mat, classes, filename):
    train_mat = pd.DataFrame(train_mat.numpy(), index=classes, columns=classes)
    test_mat = pd.DataFrame(test_mat.numpy(), index=classes, columns=classes)

    fig = plt.figure(figsize=(16, 6))

    ax = fig.add_subplot(121)
    ax = sns.heatmap(train_mat, annot=True, cmap='tab20c',
                     fmt='d', annot_kws={'size': 18})
    ax.set_title('Confusion Matrix (Train Set)', fontweight = 'bold')
    ax.set_xlabel('Predicted Classes', fontweight='bold')
    ax.set_ylabel('Actual Classes', fontweight='bold')

    ax = fig.add_subplot(122)
    ax = sns.heatmap(test_mat, annot=True, cmap='tab20c',
                     fmt='d', annot_kws={'size': 18})
    ax.set_title('Confusion Matrix (Test Set)', fontweight='bold')
    ax.set_xlabel('Predicted Classes', fontweight='bold')
    ax.set_ylabel('Actual Classes', fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(c.OUT_DIR, c.CONF_MATRIX_PATH, f'{filename}.png'))
    plt.show()
    plt.close()


def apply_mask(image, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(image)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def plot_gradcam(image, res_cam, label):
    image = deprocess_image(image)
    name_dict = {
        'Original Image': image,
        f'GradCAM ({c.SELECTED_MODEL})': apply_mask(image, res_cam),
    }

    fig = plt.figure(figsize=(6, 4))
    for i, (name, img) in enumerate(name_dict.items()):
        ax = fig.add_subplot(1, 2, i+1, xticks=[], yticks=[])
        if i:
            img = img[:, :, ::-1]
        ax.imshow(img)
        ax.set_xlabel(name, fontweight = 'bold')

    if label == 0:
        label = "Landbird"
    else:
        label = "Waterbird"

    fig.suptitle(
        f'Localization with Grad Cam : {label}',
        fontweight = 'bold', fontsize = 16
    )
    plt.tight_layout()
    fig.savefig(os.path.join(c.OUT_DIR, c.GRAD_CAM_PATH, f'grad_cam_{c.SELECTED_MODEL}_{label}.png'))
    plt.show()
    plt.close()