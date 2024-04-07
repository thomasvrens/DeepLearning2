from constants import CLASSES_RPLAN
import matplotlib.pyplot as plt
import torch
import os

def img_to_class_mask(img, n_classes):
    h, w = img.shape

    class_mask = torch.zeros(h, w, n_classes, dtype=torch.uint8)
    img_tensor = torch.tensor(img)

    for i in range(n_classes):
        class_mask[:, :, i] = torch.where(img_tensor == i, 1, 0)
    
    return class_mask

img_path = 'external_data/original'
id = 2
img = (255 * plt.imread(os.path.join(img_path, f'{id}.png'))[..., 1]).astype(int)

cls_msk = img_to_class_mask(img, 15)
print(cls_msk[:,:,12].size())
# img_to_binmap(img)