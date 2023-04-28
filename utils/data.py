import torch
import numpy as np
import albumentations as A

def get_sample(train_data, classes, ix):
    img, label = train_data[ix]
    img_np = np.array(img)
    anns = label['annotation']['object']
    if type(anns) is not list:
        anns = [anns]
    labels = np.array([classes.index(ann['name']) for ann in anns])
    bbs = [ann['bndbox'] for ann in anns]
    bbs = np.array([[int(bb['xmin']), int(bb['ymin']),int(bb['xmax'])-int(bb['xmin']),int(bb['ymax'])-int(bb['ymin'])] for bb in bbs])
    anns = (labels, bbs)
    return img_np, anns

# with coco format the bb is expected in 
# [x_min, y_min, width, height] 
def get_aug(aug, min_area=0., min_visibility=0.):
    return A.Compose(aug, bbox_params={'format': 'coco', 'min_area': min_area, 'min_visibility': min_visibility, 'label_fields': ['labels']})

def norm(bb, shape):
    # normalize bb
    # shape = (heigh, width)
    # bb = [x_min, y_min, width, height]
    h, w = shape
    return np.array([bb[0]/w, bb[1]/h, bb[2]/w, bb[3]/h])

def unnorm(bb, shape):
    # normalize bb
    # shape = (heigh, width)
    # bb = [x_min, y_min, width, height]
    h, w = shape
    return np.array([bb[0]*w, bb[1]*h, bb[2]*w, bb[3]*h])

def load_model(model, weights_path):
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint)
    return model