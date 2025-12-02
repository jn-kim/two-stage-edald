import os, random, csv, math
import pickle
from os.path import basename, splitext
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cv2
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from torchvision import transforms

from networks.ddpm import FeatureExtractorDDPM
from guided_diffusion.dist_util import dev
from .data_util import colorise_label

def last_finished_round(exp_dir: str) -> int:
    qs = [
        int(d.split("_")[0])
        for d in os.listdir(exp_dir)
        if d.endswith("_query") and d.split("_")[0].isdigit()
    ]
    return max(qs) if qs else -1

def round_status(exp_dir: str, rnd: int, last_round_saved: int) -> str:
    ckpt_dir = os.path.join(exp_dir, f"{rnd}_query")
    model_path = os.path.join(ckpt_dir, "model_0.pth")
    eval_path  = os.path.join(ckpt_dir, "evaluation_results.csv")

    has_model = os.path.isfile(model_path)
    has_eval  = os.path.isfile(eval_path)

    if not has_model:
        return "not_trained"
    if not has_eval:
        return "trained_only"
    if has_model and has_eval and last_round_saved == rnd:
        return "evaluated"

def save_last_queries(dataset, exp_dir, round_idx, seed):
    obj = dict(
        round   = round_idx,
        queries = [q.astype(bool) for q in dataset.queries]
    )
    fp = os.path.join(exp_dir, f"latest_queries_seed{seed}.pkl")
    with open(fp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_last_queries(exp_dir, seed):
    fp = os.path.join(exp_dir, f"latest_queries_seed{seed}.pkl")
    if not os.path.isfile(fp):
        return None, -1
    with open(fp, "rb") as f:
        obj = pickle.load(f)
    print(f"[RESUME] Loaded latest_queries: round {obj['round']}")
    return obj["queries"], int(obj["round"])

def load_candidates(exp_dir, round_idx, seed):
    fp = os.path.join(exp_dir,
                      f"{round_idx}_query",
                      f"local_candidates_{seed}.npy")
    return np.load(fp, allow_pickle=True)

def make_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x.mul_(2).sub_(1)
    ])
    return transform

def make_noise(args):
    if not args.share_noise:
        return None
    g = torch.Generator(device=dev()).manual_seed(args.seed)
    return torch.randn(1, 3, args.img_size, args.img_size, generator=g, device=dev())

def get_dataloader(
        args,
        batch_size: int,
        shuffle: bool,
        val: bool = False,
        query: bool = False,
        shared_subset=None
):
    from dataloader.universal import UniversalDataset
    dataset = UniversalDataset(args, val=val, query=query, transform=make_transform(), shared_subset=shared_subset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=True
    )
    return dataloader

def get_model(args):
    if args.network_name == "ddpm":
        model = FeatureExtractorDDPM(args=args)
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_seed(seed):
    print('Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def predict_labels(models, features, size, args):
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    mean_seg = None
    all_seg = []
    all_entropy = []
    seg_mode_ensemble = []
    all_probs = []

    softmax_f = nn.Softmax(dim=1)
    with torch.no_grad():
        for MODEL_NUMBER in range(len(models)):
            if args.segmentation_head == 'mlp':  
                preds = models[MODEL_NUMBER](features.float().cuda())
            prob = softmax_f(preds)
            all_probs.append(prob)

            entropy = Categorical(logits=preds).entropy()
            all_entropy.append(entropy)
            all_seg.append(preds)

            if mean_seg is None:
                mean_seg = prob
            else:
                mean_seg += prob

            y_pred_softmax = torch.log_softmax(preds, dim=1)
            _, img_seg = torch.max(y_pred_softmax, dim=1)
            img_seg = img_seg.view(size[0], size[1], size[2])
            img_seg = img_seg.cpu().detach()
            seg_mode_ensemble.append(img_seg)
        
        mean_seg = mean_seg / len(all_seg)
        full_entropy = Categorical(mean_seg).entropy()
        js = full_entropy - torch.mean(torch.stack(all_entropy), 0)
        top_k = js.sort()[0][- int(js.shape[0] / 10):].mean()

        img_seg_final = torch.stack(seg_mode_ensemble, dim=0)
        img_seg_final = torch.mode(img_seg_final, dim=0)[0]
        mean_prob = torch.mean(torch.stack(all_probs), dim=0)
    
    return img_seg_final, top_k, mean_prob # top_k: 상위 10% 픽셀의 평균 uncertainty

def norm_entropy(counts, n_classes):
    total = int(sum(counts))
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    H = -sum(p * math.log2(p) for p in probs)
    return H / math.log2(n_classes)

def spatial_coverage(mask):
    coords = np.column_stack(np.where(mask))
    if coords.shape[0] < 2:
        return 0.0
    d = np.sqrt(((coords[:, None] - coords[None]) ** 2).sum(-1))
    vals = d[~np.eye(d.shape[0], dtype=bool)]
    return float(vals.mean()) if vals.size else 0.0

def compute_query_statistics(queries, label_paths, n_classes, ignore_index):
    class_hist = np.zeros(n_classes, dtype=np.int64)
    img_pixcnt = np.zeros(len(queries), dtype=np.int64)
    unique_label_counts = []
    spatial_coverages = []

    for i, (qmask, lbl_path) in enumerate(zip(queries, label_paths)):
        if not qmask.any():
            continue
        lbl = np.load(lbl_path)
        pix_cls = lbl[qmask]
        if ignore_index is not None:
            pix_cls = pix_cls[pix_cls != ignore_index]
        img_pixcnt[i] = len(pix_cls)
        for c in pix_cls:
            class_hist[c] += 1
        if len(pix_cls) > 0:
            unique_label_counts.append(len(np.unique(pix_cls)))
        cov = spatial_coverage(qmask)
        if cov > 0:
            spatial_coverages.append(cov)

    return class_hist, img_pixcnt, unique_label_counts, spatial_coverages

def write_accumulated_query(train_loader, base_dir, args, nth_query):
    os.makedirs(base_dir, exist_ok=True)
    dataset = train_loader.dataset
    queries = dataset.queries
    labels = dataset.list_labels
    n_cls, ig_idx = args.n_classes, args.ignore_index

    class_hist, img_pixcnt, uniq, spcov = compute_query_statistics(queries, labels, n_cls, ig_idx)

    total_q = int(sum(img_pixcnt))
    expected = round(len(queries) * args.final_budget_factor) * (nth_query + 1)
    focus_ratio = img_pixcnt.max() / img_pixcnt[img_pixcnt > 0].mean() if (img_pixcnt > 0).any() else 0.0
    zero_ratio = (img_pixcnt == 0).mean()
    entropy_norm = norm_entropy(class_hist, n_cls)
    avg_uniq = np.mean(uniq) if uniq else 0.0
    avg_spcov = np.mean(spcov) if spcov else 0.0
    bincnt = np.bincount(img_pixcnt, minlength=img_pixcnt.max() + 1)

    csv_path = os.path.join(base_dir, "accumulated_query_stats.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Section", "Metric / Class / #Queried Pixels", "Value / Count / #Images"])
        writer.writerows([
            ["Summary", "Total labeled pixels", total_q],
            ["Summary", "Expected labeled pixels", expected],
            ["Summary", "Focus ratio", round(focus_ratio, 3)],
            ["Summary", "Zero-image ratio", round(zero_ratio, 3)],
            ["Summary", "Normalized entropy", round(entropy_norm, 3)],
            ["Summary", "Average #unique classes", round(avg_uniq, 3)],
            ["Summary", "Average spatial coverage", round(avg_spcov, 3)],
            []
        ])
        writer.writerows([["Image Distribution", f"{p} queried pixels", int(cnt)] for p, cnt in enumerate(bincnt) if cnt > 0])
    print(f"\n => accumulated_query_stats.csv saved for Round {nth_query}\n")


def write_round_query(args, dataloader, n_classes, nth_query, dict_queries):
    base_dir = os.path.join(args.dir_checkpoints, f"{nth_query+1}_query")
    os.makedirs(base_dir, exist_ok=True)

    ds = dataloader.dataset
    queries, labels = [], []

    for p_img, info in dict_queries.items():
        idx = ds.list_inputs.index(p_img)
        h, w = info["height"], info["width"]
        mask = np.zeros((h, w), dtype=bool)
        mask[info["y_coords"], info["x_coords"]] = True
        queries.append(mask)
        labels.append(ds.list_labels[idx])

    class_hist, img_pixcnt, uniq, spcov = compute_query_statistics(queries, labels, n_classes, args.ignore_index)

    total_px = int(sum(img_pixcnt))
    focus_ratio = img_pixcnt.max() / img_pixcnt[img_pixcnt > 0].mean() if (img_pixcnt > 0).any() else 0.0
    zero_ratio = (img_pixcnt == 0).mean()
    entropy_norm = norm_entropy(class_hist, n_classes)
    avg_uniq = np.mean(uniq) if uniq else 0.0
    avg_spcov = np.mean(spcov) if spcov else 0.0
    bincnt = np.bincount(img_pixcnt, minlength=img_pixcnt.max() + 1)

    csv_path = os.path.join(base_dir, "round_query_stats.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Section", "Metric / Class / #New Pixels", "Value / Count / #Images"])
        writer.writerows([
            ["Summary", "Total new pixels", total_px],
            ["Summary", "Focus ratio", round(focus_ratio, 3)],
            ["Summary", "Zero-image ratio", round(zero_ratio, 3)],
            ["Summary", "Normalized entropy", round(entropy_norm, 3)],
            ["Summary", "Average #unique classes", round(avg_uniq, 3)],
            ["Summary", "Average spatial coverage", round(avg_spcov, 3)],
            []
        ])
        writer.writerows([["Per-Class Distribution", str(i), int(c)] for i, c in enumerate(class_hist)])
        writer.writerow([])
        writer.writerows([["Image Distribution", f"{p} new pixels", int(cnt)] for p, cnt in enumerate(bincnt) if cnt > 0])
    print(f"\n => round_query_stats.csv saved for Round {nth_query+1}\n")

def get_img_basename(p):
    return splitext(basename(p))[0]

def visualize_mask_on_img(image_path: str, mask: np.ndarray, save_path: str):
    glow_radius = 10
    dot_radius = 3
    glow_color = (255, 0, 0)
    dot_color = (255, 0, 0)
    blur_ksize = (31, 31)
    blur_sigma = 0
    alpha = 0.4
    darken_factor = 0.8

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img_bgr.dtype != np.uint8 or img_bgr.max() <= 1.0:
        img_bgr = ((img_bgr.astype(np.float32) + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)

    img_bgr = (img_bgr.astype(np.float32) * darken_factor).astype(np.uint8)

    overlay = np.zeros_like(img_bgr)
    coords = np.argwhere(mask)

    for (row, col) in coords:
        cv2.circle(overlay, (col, row), glow_radius, glow_color, -1, cv2.LINE_AA)

    blurred = cv2.GaussianBlur(overlay, blur_ksize, blur_sigma)
    glow_result = cv2.addWeighted(blurred, alpha, img_bgr, 1 - alpha, 0)

    for (row, col) in coords:
        cv2.circle(glow_result, (col, row), dot_radius, dot_color, -1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_path.lower().endswith(".jpg") or save_path.lower().endswith(".jpeg"):
        cv2.imwrite(save_path, glow_result, [cv2.IMWRITE_JPEG_QUALITY, 100])
    else:
        cv2.imwrite(save_path, glow_result)


def visualize_mask_on_labelmap(label_map_path: str, mask: np.ndarray, save_path: str, dataset: str):
    glow_radius = 10
    dot_radius = 3
    glow_color = (255, 0, 0)
    dot_color = (255, 0, 0)
    blur_ksize = (31, 31)
    blur_sigma = 0
    alpha = 0.4
    darken_factor = 0.8

    label_map = np.load(label_map_path)
    h_label, w_label = label_map.shape

    label_img_rgb = colorise_label(label_map, dataset=dataset)
    label_img_rgb = label_img_rgb.astype(np.uint8)
    label_img_bgr = label_img_rgb[..., ::-1]
    label_img_bgr = (label_img_bgr.astype(np.float32) * darken_factor).astype(np.uint8)

    overlay = np.zeros(label_img_bgr.shape, dtype=np.uint8)
    coords = np.argwhere(mask)

    for (row, col) in coords:
        cv2.circle(overlay, (col, row), glow_radius, glow_color, -1, cv2.LINE_AA)

    blurred = cv2.GaussianBlur(overlay, blur_ksize, blur_sigma)
    glow_result = cv2.addWeighted(blurred, alpha, label_img_bgr, 1 - alpha, 0)

    for (row, col) in coords:
        cv2.circle(glow_result, (col, row), dot_radius, dot_color, -1, cv2.LINE_AA)

    if save_path.lower().endswith(".jpg") or save_path.lower().endswith(".jpeg"):
        cv2.imwrite(save_path, glow_result, [cv2.IMWRITE_JPEG_QUALITY, 100])
    else:
        cv2.imwrite(save_path, glow_result)

def visualize_accumulated_queries(dataloader, save_dir, dataset_name, max_samples=5):
    dataset = dataloader.dataset
    queries = dataset.queries

    # save_dir_label = os.path.join(save_dir, "label")
    save_dir_image = os.path.join(save_dir, "image")
    # os.makedirs(save_dir_label, exist_ok=True)
    os.makedirs(save_dir_image, exist_ok=True)

    # Collect images with queries
    images_with_queries = [(img_path, mask) for img_path, mask in zip(dataset.list_inputs, queries) if mask.any()]
    
    # Sample subset if too many (adjust max_samples to control the number of visualized images)
    if len(images_with_queries) > max_samples:
        random.seed(42)  # For reproducibility
        images_with_queries = random.sample(images_with_queries, max_samples)
    
    for image_path, mask in images_with_queries:
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # path
        # save_path_label = os.path.join(save_dir_label, f"{base_name}.png")
        save_path_image = os.path.join(save_dir_image, f"{base_name}.png")
        # label_map_path  = os.path.join(os.path.dirname(image_path), f"{base_name}.npy")

        # visualize query points
        # visualize_mask_on_labelmap(
        #     label_map_path=label_map_path,
        #     mask=mask,
        #     save_path=save_path_label,
        #     dataset=dataset_name
        # )

        visualize_mask_on_img(
            image_path=image_path,
            mask=mask,
            save_path=save_path_image
        )
