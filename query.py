import os
import numpy as np
import torch
from tqdm import tqdm  

from utils.query_utils import (
    compute_uncertainty_maps,
    apply_mc_dropout,
    encode_query,
    disable_mc_dropout,
)
from utils.herding import GlobalHerding, local_herding
from networks.ddpm import collect_features
from utils.utils import make_noise, get_model, write_round_query, get_img_basename
from utils.data_util import Visualiser 

class QuerySelector:
    def __init__(self, args, dataloader, device=torch.device("cuda:0")):
        self.args = args
        self.__dict__.update(vars(args))
        self.device = device
        self.dataloader = dataloader
        self.vis = Visualiser(args, args.dataset_name)

    def __call__(self, nth_query, model_dict):
        prev_queries = self.dataloader.dataset.queries
        local_candidates = self.dataloader.dataset.local_candidates

        feature_extractor = get_model(self.args).to(self.device)
        feature_extractor.model.eval()
        noise = make_noise(self.args)

        list_local_masks, prev_features, prev_info = [], [], []
        new_features, new_info = [], []

        xx, yy = [], []
        local_candidates = [np.asarray(arr, dtype=np.int64) for arr in local_candidates]
        
        image_loop = tqdm(
            enumerate(self.dataloader),
            total=len(self.dataloader.dataset.list_labels),
            desc="Local Herding (All Images)",
            leave=True
        )

        total_selected = 0
        expected_total = len(self.dataloader.dataset.list_labels) * self.args.local_budget

        for image_index, data in image_loop:
            x = data["x"].to(self.device)
            y = data["y"].squeeze(0).cpu().numpy()
            height, width = y.shape
            xx.append(x.squeeze(0).cpu())
            yy.append(torch.from_numpy(y))

            prev_mask = prev_queries[image_index]
            already_selected = np.where(prev_mask.flatten())[0]
            valid_pixels = np.setdiff1d(local_candidates[image_index], already_selected)
            valid_pixels = valid_pixels.astype(np.int64)

            feats, feature_dim = collect_features(
                self.args,
                feature_extractor(x, noise=noise, forced_steps=self.steps)
            )
            feats = feats.unsqueeze(0).permute(1, 0, 2, 3).reshape(feature_dim, -1).permute(1, 0)

            if already_selected.size > 0:
                prev_features.append(feats[already_selected].cpu().numpy())
                prev_info.extend((image_index, p) for p in already_selected)

            mask_available = np.zeros(height * width, bool)
            mask_available[valid_pixels] = True
            candidate_features = feats[mask_available]

            progress_desc = f"Local Herding({image_index+1}/{len(self.dataloader.dataset.list_labels)})"

            selected_local, _, init_idx, init_cov = local_herding(
                candidate_features, budget=self.args.local_budget,
                sigma="auto", batch_size=4096, device=self.device,
                disable_tqdm=False, progress_desc=progress_desc
            )

            num_selected = len(selected_local)
            total_selected += num_selected
            percent_done = (total_selected / expected_total) * 100

            image_loop.set_postfix_str(
                f"Image {image_index+1}/{len(self.dataloader.dataset.list_labels)}, "
                f"InitIdx={init_idx}, InitCov={init_cov:.4f}, "
                f"Candidates={len(valid_pixels)}, "
                f"Selected={num_selected}, "
                f"TotalSelected={total_selected}/{expected_total} ({percent_done:.1f}%)"
            )

            selected_pixels = valid_pixels[selected_local]

            local_mask_flat = np.zeros(height * width, bool)
            local_mask_flat[selected_pixels] = True
            local_mask = local_mask_flat.reshape(height, width)
            list_local_masks.append(local_mask)

            image_path = self.dataloader.dataset.list_inputs[image_index]
            
            new_features.append(candidate_features[selected_local].cpu().numpy())
            new_info.extend((image_index, p) for p in selected_pixels)

        print()
        prev_features = np.concatenate(prev_features, 0) if prev_features else np.empty((0, feature_dim), np.float32)
        new_features = np.concatenate(new_features, 0) if new_features else np.empty((0, feature_dim), np.float32)
        num_prev = len(prev_features)
        num_new = len(new_features)

        all_features = np.concatenate([prev_features, new_features], 0)
        
        herding_budget = max(1, round(num_new * self.keep_global))
        
        selector = GlobalHerding(kernel_fn=None, sigma="auto", device=self.device)
        selected_global_indices = selector.greedy_select(
            torch.tensor(all_features, device=self.device),
            budget=herding_budget,
            num_prev=num_prev
        )
        print()

        final_global_masks = [np.zeros_like(list_local_masks[0], bool) for _ in range(len(self.dataloader))]
        for idx in selected_global_indices:
            img_idx, pix_idx = (prev_info + new_info)[idx]
            final_global_masks[img_idx].flat[pix_idx] = True
        
        relevant_indices = [i for i, mask in enumerate(final_global_masks) if mask.any()]
        
        feature_extractor_uc, classifier_uc = apply_mc_dropout(self.args, model_dict, self.device)
        uncertainty_maps, prediction_maps = compute_uncertainty_maps(
            self.dataloader, 
            feature_extractor_uc, 
            classifier_uc,
            self.args, 
            self.device,
            relevant_indices=relevant_indices
        )
        disable_mc_dropout(feature_extractor_uc, classifier_uc)

        image_height, image_width = uncertainty_maps[0].shape
        selected_pixels_with_scores = []

        vis_dir = os.path.join(self.args.dir_checkpoints, f"{nth_query+1}_query", "uc_map", "vis")
        os.makedirs(vis_dir, exist_ok=True)

        for img_i, (uc_map, mask, pred) in enumerate(zip(uncertainty_maps, final_global_masks, prediction_maps)):
            flat_uc_map = uc_map.flatten()
            for pix in np.where(mask.flatten())[0]:
                selected_pixels_with_scores.append((img_i, pix, float(flat_uc_map[pix])))

            img_base = get_img_basename(self.dataloader.dataset.list_inputs[img_i])
            
            if (img_i % 100) == 0:
                vis_fp = os.path.join(vis_dir, f"{img_base}.png")
                dict_tensors = {
                    "input": xx[img_i],
                    "target": yy[img_i],
                    "pred": pred,
                    self.uncertainty: torch.from_numpy(uc_map).cpu(),
                }
                self.vis(dict_tensors, fp=vis_fp)
            
        selected_pixels_with_scores.sort(
            key=lambda t: t[2], reverse=self.uncertainty != "margin_sampling"
        )

        final_budget = round(self.args.final_budget_factor * len(self.dataloader))
        refined_masks = [np.zeros((image_height, image_width), bool) for _ in range(len(self.dataloader))]
        for img_i, pix_i, _ in selected_pixels_with_scores[:final_budget]:
            refined_masks[img_i].flat[pix_i] = True

        dict_queries        = {}
        dict_global_queries = {}

        for i, mask in enumerate(refined_masks):
            image_path = self.dataloader.dataset.list_inputs[i]
            img_base = get_img_basename(image_path)

            query_info = encode_query(image_path, size=mask.shape, query=mask)
            dict_queries[image_path] = query_info[image_path]

        for i, mask in enumerate(final_global_masks):
            image_path = self.dataloader.dataset.list_inputs[i]
            global_info = encode_query(image_path, size=mask.shape, query=mask)
            dict_global_queries[image_path] = global_info[image_path]

        write_round_query(self.args, self.dataloader, self.args.n_classes, nth_query, dict_queries)
        self.dataloader.dataset.label_queries(dict_queries, dict_global_queries)
        return dict_queries, dict_global_queries
