import os
from pathlib import Path
from typing import List, Tuple
import random
from tqdm import tqdm

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from .base import BaseDataset
from utils.herding import sample_local_candidates, local_herding, GlobalHerding
from utils.utils import get_model
from networks.ddpm import collect_features

class UniversalDataset(BaseDataset):
    def __init__(self, args, val=False, query=False, transform=None, shared_subset=None):
        super().__init__(transform=transform)
        self.args = args
        self.__dict__.update(vars(args))
        self.img_ext = args.img_ext
        self.data_usage = 100 if val else args.data_usage
        
        self.val = val
        self.query = query
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.queries = self.global_queries = self.local_candidates = None

        self.path_queries = self._get_query_path()
        mode = self._get_mode(val)
        dataset_path = self._get_dataset_path(mode)
        
        self._is_resume = (
            args.resume
            and shared_subset is not None
            and os.path.isfile(self.path_queries)
        )
        
        matched = self._get_matched_files(dataset_path)
        assert matched, f"No valid dataset found in {dataset_path}!"

        if shared_subset:
            self.list_inputs, self.list_labels, self.queries, self.global_queries, self.local_candidates = shared_subset
        else:
            self._prepare_dataset_split(matched, val, query)
        
        # Train, Query dataloader
        if not val:
            split_dir = os.path.join(self.dir_checkpoints)
            os.makedirs(split_dir, exist_ok=True)
            list_fp = os.path.join(split_dir, f"list_inputs_seed{self.seed}.txt")

            if os.path.isfile(list_fp):
                with open(list_fp) as f:
                    fixed_inputs = [ln.strip() for ln in f]
                self.list_inputs = fixed_inputs
                
                self.list_labels = [
                    self.label_map[Path(p).stem]
                    for p in fixed_inputs
                ]
            else:
                with open(list_fp, "w") as f:
                    for p in self.list_inputs:
                        f.write(p + "\n")
                print(f"Saved list_inputs_seed{self.seed}.txt.")
        
        # Generate round 0 queries (only when not resuming train dataset)
        if not val and args.final_budget_factor != 0 and not self._is_resume:
            print("Generating initial queries...")
            
            np.random.seed(self.seed)
            
            # Check if regeneration is needed
            cand_path = self.path_queries.parent / f"local_candidates_{self.seed}.npy"

            if not self.path_queries.is_file() or not cand_path.is_file():
                regenerate = True
                print("Initial query files not found, will regenerate.")
            else:
                queries_data = np.load(self.path_queries, allow_pickle=True)
                
                if queries_data.shape[0] != len(self.list_labels):
                    regenerate = True
                    print("Number of images has changed, will regenerate.")
                else:
                    regenerate = False
                    print("Found valid query files, loading data.")
                    self.queries = queries_data
                    self.local_candidates = np.load(cand_path, allow_pickle=True)
                    if self.global_queries is None:
                        self.global_queries = np.array(self.queries, copy=True)
            
            if regenerate:
                print("No predefined queries found. Regenerating...")

                feature_extractor = get_model(self.args).to(self.device)
                local_masks, features_list, candidate_list = [], [], []

                image_loop = tqdm(
                    enumerate(zip(self.list_inputs, self.list_labels)),
                    total=len(self.list_labels),
                    desc="Local Herding (All Images)",
                    leave=True,
                    mininterval=0.5
                )
                total_selected = 0
                expected_total = len(self.list_labels) * self.local_budget_init
                for i, (img_path, label_path) in image_loop:
                    img = TF.to_tensor(Image.open(img_path).convert("RGB"))[None].to(self.device)
                    label = np.load(label_path).astype("uint8")
                    features, _ = collect_features(self.args, feature_extractor(img, noise=None)) # dtype: fp16
                    features = features.squeeze(0).permute(1, 2, 0).reshape(-1, features.shape[0])
                    cand_mask = sample_local_candidates(self.args, label).flatten()
                    cand_idx = np.where(cand_mask)[0]

                    image_loop.set_postfix_str(
                        f"Image {i+1}/{len(self.list_labels)}, Candidates={len(cand_idx)}"
                    )

                    cand_features = features[cand_mask]
                    selected_idx, _, init_idx, init_cov = local_herding(
                        cand_features, budget=self.local_budget_init,
                        kernel_fn=None, sigma='auto',
                        batch_size=4096, device=self.device,
                        disable_tqdm=False, progress_desc=f"Img {i+1}"
                    )
                    num_selected = len(selected_idx)
                    total_selected += num_selected
                    image_loop.set_postfix_str(
                        f"Image {i+1}/{len(self.list_labels)}, "
                        f"InitIdx={init_idx}, InitCov={init_cov:.4f}, "
                        f"Candidates={len(cand_idx)}, "
                        f"Selected={num_selected}, TotalSelected={total_selected}/{expected_total}"
                    )
                    query_mask = np.zeros(label.size, dtype=bool)
                    query_mask[cand_idx[selected_idx]] = True
                    local_masks.append(query_mask.reshape(label.shape))
                    features_list.append(cand_features[selected_idx].cpu().numpy())
                    candidate_list.append(cand_idx)
                
                # Save local candidates
                save_dir = os.path.dirname(self.path_queries)
                os.makedirs(save_dir, exist_ok=True)
                np.save(
                    os.path.join(save_dir, f"local_candidates_{self.seed}.npy"),
                    np.array(candidate_list, dtype=object)
                )
                # Global Herding
                global_feats, global_info = [], []
                for i, mask in enumerate(local_masks):
                    for j, pix_idx in enumerate(np.where(mask.flatten())[0]):
                        global_feats.append(features_list[i][j])
                        global_info.append((i, pix_idx))

                global_feats_tensor = torch.tensor(np.array(global_feats)) # Convert to fp32
                final_budget = round(self.final_budget_factor * len(self.list_labels))
                print("\nFinal Budget:", final_budget)
                
                selector = GlobalHerding(kernel_fn=None, sigma='auto', device=self.device)
                selected = selector.greedy_select(global_feats_tensor, final_budget)

                queries = [np.zeros(local_masks[0].size, dtype=bool) for _ in range(len(self.list_labels))]
                for idx in selected:
                    i, pix = global_info[idx]
                    queries[i][pix] = True

                self.queries = self.global_queries = np.array([q.reshape(local_masks[0].shape) for q in queries], dtype=bool)
                self.local_candidates = candidate_list
                np.save(self.path_queries, self.queries)
            
            print(f"Round 0: Total labelled pixels for initial query: {self.queries.sum()}")

    def __len__(self):
        return len(self.list_inputs)

    def _get_mode(self, val):
        return {"ade": "test", "camvid": "test", "cityscapes": "val", "pascal": "val"}.get(self.dataset_name, "val") if val else "train"

    def _get_dataset_path(self, mode):
        return os.path.join(self.dir_dataset, mode)

    def _prepare_dataset_split(self, matched, val, query):
        random.seed(self.seed)
        random.shuffle(matched)

        if not val:
            n = round(len(matched) * (self.data_usage / 100.0))
            matched = matched[:n]

        self.list_inputs, self.list_labels = map(list, zip(*matched)) if matched else ([], [])
        phase = "Train" if not val and not query else "Query" if query else "Val"
        print(f"\n[{phase} Dataset] {self.data_usage}% used: {len(self.list_inputs)} samples\n")
    
    def _get_matched_files(self, dataset_path: str) -> List[Tuple[str, str]]:
        img_paths = sorted(Path(dataset_path).glob(f'*.{self.img_ext}'))
        self.label_map = {p.stem: str(p) for p in Path(dataset_path).glob('*.npy')}
        
        matched_files = []
        for img_path in img_paths:
            if img_path.stem in self.label_map:
                matched_files.append((str(img_path), self.label_map[img_path.stem]))
        
        return matched_files
    
    def _get_query_path(self):
            return Path(self.args.dir_checkpoints) / "0_query" / f"init_labelled_pixels_{self.args.seed}.npy"