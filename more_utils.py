from constants import CLASSES_RPLAN
import matplotlib.pyplot as plt
import torch
import os

def img_to_class_mask(img, n_classes):
    class_img = (255 * img)[..., 1].astype(int)
    h, w = class_img.shape

    class_mask = torch.zeros(h, w, n_classes, dtype=torch.uint8)
    img_tensor = torch.tensor(class_img)

    for i in range(n_classes):
        class_mask[:, :, i] = torch.where(img_tensor == i, 1, 0)
    
    return class_mask