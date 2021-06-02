import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import holemask, onlycenter
from PIL import Image

if __name__ == "__main__":
    chexpert_path = "/CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg"
    image = Image.open(chexpert_path)
    image = image.resize((2800, 2800))
    image = np.array(image)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
    ax1.imshow(image, cmap="gray")
    ax2.imshow(np.squeeze(holemask(image, h=int(2800*.75), w=int(2800*.75), SIDE=2800, is_torch=False), axis=(0,1)), cmap="gray")
    ax3.imshow(np.squeeze(onlycenter(image, h=int(2800*.75), w=int(2800*.75), SIDE=2800, is_torch=False), axis=(0,1)), cmap="gray")
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    plt.savefig('masking.pdf', bbox_inches='tight')
    # image = cv2.imread(chexpert_path, 0)
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # # shape is (2800, 3408)
    # ax1.imshow(image, cmap='gray')
    # ax1.imshow(image, cmap='gray')
    
