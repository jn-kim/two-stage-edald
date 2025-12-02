import os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.cm as cm
import torch
from PIL import Image

class Visualiser:
    def __init__(self, args, dataset_name):
        self.args = args
        self.palette = {
            "camvid": palette_cv,
            "cityscapes": palette_cs,
            "ade": palette_ade,
            "pascal": palette_pascal
        }.get(dataset_name, palette_cv)

    def _preprocess(self, tensor, seg, downsample=2, key=None):
        tensor = tensor.cpu().detach()

        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        if tensor.ndim == 3 and tensor.shape[0] in [1, 3, 4]:
            tensor = tensor.permute(1, 2, 0)
        h, w = tensor.shape[:2]

        if seg:
            tensor_flat = torch.clamp(tensor.flatten(), 0, 255)
            grid = torch.zeros((h * w, 3), dtype=torch.uint8)
            for i in range(h * w):
                label_idx = int(tensor_flat[i].item())
                grid[i] = torch.tensor(self.palette[label_idx % len(self.palette)], dtype=torch.uint8)
            arr = grid.view(h, w, 3).numpy().astype(np.uint8)
        # uncertainty map visualization
        else:
            np_arr = tensor.numpy()

            if key == "input":
                np_arr = np.clip((np_arr + 1.0) / 2.0, 0, 1)
                arr = (np_arr * 255).astype(np.uint8)

            elif key in {
                "bald", "power_bald", "balentacq", "entropy_bald",
                "entropy", "least_confidence", "confidence", "margin_sampling"
            }:
                np_arr = np.clip(np_arr, 0, None)
                np_arr = np.log1p(np_arr)
                np_arr = np.power(np_arr, 0.7)

                if key in {"margin", "margin_sampling"}:
                    np_arr = 1.0 - (np_arr - np_arr.min()) / (np_arr.max() - np_arr.min() + 1e-8)
                else:
                    np_arr = (np_arr - np_arr.min()) / (np_arr.max() - np_arr.min() + 1e-8)

                vmax = np.percentile(np_arr, 99)
                np_arr = np.clip(np_arr, 0, vmax) / (vmax + 1e-8)

                heatmap = cm.jet(np_arr)[:, :, :3]
                arr = (heatmap * 255).astype(np.uint8)

            else:
                np_arr = np.clip(np_arr, 0, None)
                np_arr = (np_arr - np_arr.min()) / (np_arr.max() - np_arr.min() + 1e-8)
                arr = (np_arr * 255).astype(np.uint8)

            if arr.ndim == 3 and arr.shape[0] in [1, 3, 4]:
                arr = np.transpose(arr, (1, 2, 0))
            elif arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr.squeeze(-1)

        if arr.ndim > 3:
            arr = np.squeeze(arr)

        return Image.fromarray(arr)

    @staticmethod
    def _make_grid(list_imgs):
        total_width = sum(img.width for img in list_imgs)
        height = list_imgs[0].height
        grid = Image.new("RGB", (total_width, height))
        x_offset = 0
        for img in list_imgs:
            grid.paste(img, (x_offset, 0))
            x_offset += img.width
        return grid

    def __call__(self, dict_tensors, fp='', show=False):
        list_imgs = []

        input_tensor = dict_tensors['input']
        if input_tensor.ndim == 3 and input_tensor.shape[0] == 3:
            input_tensor = input_tensor.permute(1, 2, 0)
        list_imgs.append(self._preprocess(input_tensor, seg=False, key="input"))

        if 'target' in dict_tensors and dict_tensors['target'] is not None:
            list_imgs.append(self._preprocess(dict_tensors['target'], seg=True, key="target"))

        list_imgs.append(self._preprocess(dict_tensors['pred'], seg=True, key="pred"))
        
        for key in [
            'least_confidence', 'confidence', 'margin_sampling', 'margin',
            'entropy', 'bald', 'power_bald', 'balentacq', 'kl_div', 'cosine_sim',
            "entropy_bald", "entropy_power_bald", "entropy_balentacq"
        ]:
            if key in dict_tensors:
                list_imgs.append(self._preprocess(dict_tensors[key], seg=False, key=key))

        img = self._make_grid(list_imgs)

        if fp:
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            img.save(fp, dpi=(300, 300))

        if show:
            img.show()

def colorise_label(arr, dataset="camvid"):
    assert len(arr.shape) == 2, arr.shape
    assert dataset in ["camvid", "cityscapes", "ade", "pascal"], dataset
    if dataset == "camvid":
        global palette_cv
        palette = palette_cv

    elif dataset == "cityscapes":
        global palette_cs
        palette = palette_cs
        
    elif dataset == "ade":
        global palette_ade
        palette = palette_ade
    
    elif dataset == "pascal":
        global palette_pascal
        palette = palette_pascal

    grid = np.empty((3, *arr.shape), dtype=np.uint8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            grid[:, i, j] = palette[arr[i, j]]

    return np.transpose(grid, (1, 2, 0))


palette_cv = {
    0: (128, 128, 128),
    1: (128, 0, 0),
    2: (192, 192, 128),
    3: (128, 64, 128),
    4: (0, 0, 192),
    5: (128, 128, 0),
    6: (192, 128, 128),
    7: (64, 64, 128),
    8: (64, 0, 128),
    9: (64, 64, 0),
    10: (0, 128, 192),
    255: (0, 0, 0)
}

palette_cs = {
    0: (128, 64, 128),
    1: (244, 35, 232),
    2: (70, 70, 70),
    3: (102, 102, 156),
    4: (190, 153, 153),
    5: (153, 153, 153),
    6: (250, 170, 30),
    7: (220, 220, 0),
    8: (107, 142, 35),
    9: (152, 251, 152),
    10: (70, 130, 180),
    11: (220, 20, 60),
    12: (255, 0, 0),
    13: (0, 0, 142),
    14: (0, 0, 70),
    15: (0, 60, 100),
    16: (0, 80, 100),
    17: (0, 0, 230),
    18: (119, 11, 32),
    255: (0, 0, 0)
}

palette_ade = {
    0: (240, 156, 206),
    1: (69, 88, 93),
    2: (240, 49, 184),
    3: (27, 107, 126),
    4: (50, 82, 241),
    5: (54, 250, 147),
    6: (156, 213, 3),
    7: (176, 108, 79),
    8: (251, 150, 149),
    9: (66, 51, 34),
    10: (210, 97, 53),
    11: (30, 53, 102),
    12: (232, 164, 118),
    13: (204, 150, 17),
    14: (101, 86, 178),
    15: (249, 20, 213),
    16: (54, 35, 82),
    17: (157, 68, 216),
    18: (58, 161, 73),
    19: (174, 67, 67),
    20: (193, 181, 78),
    21: (169, 60, 178),
    22: (220, 204, 166),
    23: (4, 127, 85),
    24: (245, 106, 216),
    25: (222, 172, 168),
    26: (84, 148, 105),
    27: (137, 220, 89),
    28: (68, 252, 126),
    29: (29, 193, 187),
    255: (255, 255, 255)
}

palette_pascal = {
    0: (0, 128, 0),
    1: (128, 192, 128),
    2: (192, 64, 0),
    3: (192, 192, 128),
    4: (0, 128, 64),
    5: (192, 0, 192),
    6: (192, 192, 64),
    7: (160, 0, 0),
    8: (96, 0, 0),
    9: (32, 128, 64),
    10: (192, 32, 160),
    11: (160, 64, 64),
    12: (224, 160, 128),
    13: (0, 128, 32),
    14: (64, 64, 160),
    15: (0, 64, 224),
    16: (32, 64, 96),
    17: (32, 64, 96),
    18: (0, 32, 96),
    19: (192, 160, 96),
    20: (96, 0, 96),
    21: (192, 224, 64),
    22: (192, 96, 192),
    23: (32, 0, 160),
    24: (64, 0, 192),
    25: (0, 32, 224),
    26: (192, 96, 224),
    27: (128, 128, 32),
    28: (64, 96, 96),
    29: (64, 224, 128),
    30: (192, 224, 160),
    31: (96, 96, 128),
    32: (32, 0, 128),
    255: (255, 255, 255)
}

ade_bedroom_30_class = ["wall", 
                        "bed", 
                        "floor", 
                        "table", 
                        "lamp", 
                        "ceiling", 
                        "painting", 
                        "windowpane",
                        "pillow", 
                        "curtain", 
                        "cushion", 
                        "door", 
                        "chair", 
                        "cabinet", 
                        "chest", 
                        "mirror", 
                        "rug", 
                        "armchair", 
                        "book", 
                        "sconce", 
                        "plant", 
                        "wardrobe", 
                        "clock", 
                        "light", 
                        "flower", 
                        "vase", 
                        "fan", 
                        "box", 
                        "shelf", 
                        "television"]

cityscapes_class = [
    'road',
    'sidewalk',
    'building',
    'wall',
    'fence',
    'pole',
    'traffic light',
    'traffic sign',
    'vegetation',
    'terrain',
    'sky',
    'person',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle'
]


camvid_class = ['sky', 
                'building', 
                'pole', 
                'road', 
                'pavement', 
                'tree', 
                'sign_symbol', 
                'fence', 
                'car', 
                'pedestrian', 
                'bicyclist']


pascal_class = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'table',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
    'sky',
    'grass',
    'ground',
    'road',
    'building',
    'tree',
    'water',
    'mountain',
    'wall',
    'floor',
    'track',
    'keyboard',
    'ceiling'
]
