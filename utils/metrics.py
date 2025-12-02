import os
import csv
import numpy as np
from collections import Counter
from PIL import Image
import torch
from .data_util import ade_bedroom_30_class, camvid_class, cityscapes_class, pascal_class

class AverageMeter(object):
    def __init__(self):
        self.initialized = False
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def reset(self):
        self.initialized = False
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True
    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)
    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count
    @property
    def value(self):
        return self.val
    @property
    def average(self):
        return np.round(self.avg, 5)

class RunningScore(object):
    def __init__(self, n_classes, ignore_index=None, dataset_name="ade", category=None):
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.dataset_name = dataset_name
        self.class_names = self.get_class_names(dataset_name)
        self.category = category
        self.reset()

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self._update_pixelwise(lt.flatten(), lp.flatten())
 
    def _update_pixelwise(self, label_true, label_pred):
        valid_mask = label_true != self.ignore_index
        for class_id in range(self.n_classes):
            if self.ignore_index is not None and class_id == self.ignore_index:
                continue

            pred_mask = (label_pred == class_id) & valid_mask
            gt_mask = (label_true == class_id) & valid_mask

            self.intersections[class_id] += (pred_mask & gt_mask).sum()
            self.unions[class_id] += (pred_mask | gt_mask).sum()
            self.correct_per_class[class_id] += (pred_mask & gt_mask).sum()
            self.total_per_class[class_id] += gt_mask.sum()

        self.correct_pixels += ((label_pred == label_true) & valid_mask).sum()
        self.total_pixels += valid_mask.sum()

    def get_scores(self, return_classwise=True):
        ious, class_accs = [], []
        for class_id in range(self.n_classes):
            if self.ignore_index is not None and class_id == self.ignore_index:
                continue
            iou = self.intersections[class_id] / (1e-8 + self.unions[class_id])
            acc = self.correct_per_class[class_id] / (1e-8 + self.total_per_class[class_id])
            ious.append(iou)
            class_accs.append(acc)

        miou = np.mean(ious)
        mean_class_acc = np.mean(class_accs)
        pixel_acc = self.correct_pixels / (1e-8 + self.total_pixels)

        if return_classwise:
            return {
                "Pixel Acc": pixel_acc,
                "Mean IoU": miou,
                "Mean Class Acc": mean_class_acc
            }, self.class_names, ious, class_accs
        return {
            "Pixel Acc": pixel_acc,
            "Mean IoU": miou,
            "Mean Class Acc": mean_class_acc
        }

    def reset(self):
        self.unions = Counter()
        self.intersections = Counter()
        self.correct_pixels = 0
        self.total_pixels = 0
        self.correct_per_class = Counter()
        self.total_per_class = Counter()

    def get_class_names(self, category):
        if category == 'ade':
            return ade_bedroom_30_class
        elif category == 'camvid':
            return camvid_class
        elif category == 'cityscapes':
            return cityscapes_class
        elif category =="pascal":
            return pascal_class
        else:
            raise ValueError(f"Unknown dataset category: {category}")

    def oht_to_scalar(self, y_pred):
        y_pred_softmax = torch.log_softmax(y_pred, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
        return y_pred_tags

    def colorize_mask(self, mask, palette):
        new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
        new_mask.putpalette(palette)
        return np.array(new_mask.convert('RGB'))

    def to_labels(self, masks, palette):
        results = np.zeros((len(masks), 256, 256), dtype=np.int32)
        label = 0
        for color in palette:
            idxs = np.where((masks == color).all(-1))
            results[idxs] = label
            label += 1
        return results

def multi_acc(y_pred, y_test, ignore_index=255):
    _, y_pred_tags = torch.max(y_pred, dim=1)
    valid_mask = (y_test != ignore_index)
    correct_pred = (y_pred_tags[valid_mask] == y_test[valid_mask]).float()
    acc = correct_pred.sum() / valid_mask.sum()
    acc = acc * 100.0
    return acc

def batch_miou(pred, label, n_cls):
    B, H, W = pred.shape
    miou = torch.zeros(B, device=pred.device)
    for c in range(n_cls):
        p = (pred == c)
        g = (label == c)
        inter = (p & g).sum(dim=(1, 2)).float()
        union = p.sum(dim=(1, 2)) + g.sum(dim=(1, 2)) - inter
        valid = union > 0
        miou += torch.where(valid, inter / union, torch.zeros_like(inter))
    return miou / n_cls

def save_scores_to_csv(csv_path, class_names, class_ious, class_accs, miou, mean_class_acc, pixel_acc):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Class Name", "IoU", "Class-wise Pixel Accuracy", "Pixel Accuracy"])

        for name, iou, acc in zip(class_names, class_ious, class_accs):
            writer.writerow([name, f"{iou:.4f}", f"{acc:.4f}", ""])

        writer.writerow(["Mean", f"{miou:.4f}", f"{mean_class_acc:.4f}", f"{pixel_acc:.4f}"])
