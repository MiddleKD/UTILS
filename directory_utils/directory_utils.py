import os
import shutil
import numpy as np
from PIL import Image

def copy_and_paste(path, target_path):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.copy(path, target_path)
    return True

def open_image(path):
    img = np.array(Image.open(path).resize((128,128)))
    return img

def save_image(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)
    return True