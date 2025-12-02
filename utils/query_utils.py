import os
import random
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from networks.ddpm import collect_batch_features
from .acquisition_functions import (
    bald_acquisition_function,
    balentacq_acquisition_function,
    power_bald_acquisition_function,
)
from .utils import get_model, make_noise, get_img_basename

def encode_query(p_img: str, size: Tuple[int, int], query: np.ndarray,) -> Dict[str, dict]:
    
    y_coords, x_coords = np.where(query)
    
    return {
        p_img: {
            "height": size[0],
            "width": size[1],
            "x_coords": x_coords,
            "y_coords": y_coords,
        }
    }
    
def decode_queries(
    encoded_query: Dict[str, dict],
    list_inputs: List[str],
) -> List[np.ndarray]:
    
    def _decode_single(query_info: dict) -> np.ndarray:
        q = np.zeros((query_info["height"], query_info["width"]), dtype=bool)
        for y, x in zip(query_info["y_coords"], query_info["x_coords"]):
            q[y, x] = True
        return q

    result = []
    for path in list_inputs:
        assert path in encoded_query, f"[decode_queries] Missing query for image: {path}"
        result.append(_decode_single(encoded_query[path]))
    return result

def is_bald_method(args) -> bool:
    """
    Check if using BALD (Bayesian Active Learning by Disagreement) method.
    BALD uses MC-Dropout for Bayesian approximation.
    """
    return (
        args.uncertainty in {
            "bald", "power_bald", "balentacq", "entropy_bald"
        }
        and args.use_mc_dropout
    )

def is_dald_method(args) -> bool:
    """
    Check if using DALD (Diffusion Active Learning by Disagreement) method.
    DALD uses diffusion stochastic sampling for Bayesian approximation.
    """
    return (
        args.uncertainty in {
            "dald", "power_dald", "entropy_dald"
        }
        and args.use_diffusion_stochastic
    )

def _set_dropout_state(model: nn.Module, train: bool):
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d,
                          nn.Dropout2d, nn.Dropout3d)):
            m.train(train)

def disable_mc_dropout(feature_extractor, classifiers):
    _set_dropout_state(feature_extractor, False)
    for clf in classifiers:
        _set_dropout_state(clf, False)

def apply_mc_dropout(args, model_dict, device):
    feature_extractor = get_model(args).to(device)

    concat_clf = model_dict["concat"]
    raw_blocks = model_dict["blocks"]
    block_clfs = [
        (blk["classifiers"][0] if isinstance(blk, dict) else blk)
        for blk in raw_blocks
    ]
    concat_clf = concat_clf.to(device)
    block_clfs = [blk.to(device) for blk in block_clfs]

    classifiers = [concat_clf] + block_clfs

    if is_bald_method(args):
        _set_dropout_state(feature_extractor, True)
        _set_dropout_state(concat_clf, True)
        for blk in block_clfs:
            blk.eval()
    else:
        feature_extractor.model.eval()
        for clf in classifiers:
            clf.eval()

    return feature_extractor, classifiers


class UncertaintySampler:
    def __init__(self, uncertainty: str, args=None):
        self.uncertainty = uncertainty
        self.args = args
        self.hybrid_tail = "_" + uncertainty.split("_", 1)[1] \
            if uncertainty.startswith("entropy_") else None
         
    def _entropy(self, prob):
        return (-prob * torch.log(prob + 1e-8)).sum(dim=1)

    def _least_confidence(self, prob):
        return 1.0 - prob.max(dim=1)[0]

    def _margin_sampling(self, prob):
        top2 = prob.topk(k=2, dim=1).values
        return (top2[:, 0] - top2[:, 1]).abs()

    def _random(self, prob):
        b, _, h, w = prob.shape
        return torch.rand(b, h, w, device=prob.device)

    def _bald(self, logits):
        b, k, c, h, w = logits.shape
        flat = logits.permute(0, 3, 4, 1, 2).reshape(-1, k, c) # [b*h*w, k, c]
        return bald_acquisition_function(flat).view(b, h, w) # [b,h,w]

    def _power_bald(self, logits):
        b, k, c, h, w = logits.shape
        flat = logits.permute(0, 3, 4, 1, 2).reshape(-1, k, c)
        return power_bald_acquisition_function(flat).view(b, h, w)

    def _balentacq(self, logits):
        b, k, c, h, w = logits.shape
        flat = logits.permute(0, 3, 4, 1, 2).reshape(-1, k, c)
        return balentacq_acquisition_function(flat).view(b, h, w)

    def _dald(self, logits):
        return self._bald(logits)

    def _power_dald(self, logits):
        return self._power_bald(logits)

    def _entropy_hybrid(self, hybrid_dict, img_tag):
        ent_map   = self._entropy(hybrid_dict["entropy_probs"])
        second_fn = getattr(self, self.hybrid_tail)
        second_map = second_fn(hybrid_dict["bald_logits"])

        combined = ent_map + second_map
        return combined

    def __call__(self, x, *, img_tag):
        if self.hybrid_tail is not None:
            return self._entropy_hybrid(x, img_tag)

        fn    = getattr(self, f"_{self.uncertainty}")
        score = fn(x)
        return score


def _forward_logits(imgs, feature_extractor, concat_clf, args, diff_inputs, h, w, use_bald, K):
    feats = feature_extractor(
        imgs,
        noise=diff_inputs.get("noise", None),
        forced_steps=diff_inputs.get("forced_steps", None),
    )
    # Collect and concatenate features from multiple blocks: [B, D, H, W]
    collected, _ = collect_batch_features(args, feats)
    
    dim = collected.shape[1]
    flat = collected.permute(0, 2, 3, 1).reshape(-1, dim)
    logits = concat_clf(flat.float()).reshape(K, h, w, args.n_classes).permute(0, 3, 1, 2)
    probs = torch.log_softmax(logits, dim=1) if use_bald else torch.softmax(logits, dim=1)
    return logits, probs

def _append_probs_as_samples(samples: List[torch.Tensor], probs: torch.Tensor):
    for b in range(len(probs)):
        samples.append(probs[b].unsqueeze(0).unsqueeze(0))

def compute_uncertainty_maps(dataloader, feature_extractor, classifiers, args, device, relevant_indices: Optional[list] = None):
    print("\nComputing uncertainty maps")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    sampler = UncertaintySampler(
        args.uncertainty,
        args=args,
    )

    base_noise  = make_noise(args)
    concat_clf  = classifiers[0]

    # BALD/DALD-based methods: use mutual information for uncertainty estimation
    BALD_BASED  = {"bald", "power_bald", "balentacq", "dald", "power_dald"}
    use_bald    = sampler.uncertainty in BALD_BASED
    is_entropy  = args.uncertainty.startswith("entropy_")

    relevant_indices = set(relevant_indices) if relevant_indices else None
    uc_maps, pred_maps = [], []
    
    with tqdm(dataloader, desc="Images", position=0, dynamic_ncols=True, 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as outer_bar:
        for idx, batch in enumerate(outer_bar):
            img_tag = get_img_basename(dataloader.dataset.list_inputs[idx])
            if relevant_indices and idx not in relevant_indices:
                fill_val = 1e9 if args.uncertainty == "margin_sampling" else 0.0
                img_size = args.img_size
                uc_maps.append(np.full((img_size, img_size), fill_val, np.float32))
                pred_maps.append(torch.zeros((img_size, img_size), dtype=torch.int64))
                continue

            with torch.no_grad():
                x = batch["x"].to(device)
                _, _, h, w = x.shape
                logit_samples, pred = [], None

                diff_fixed = {
                    "noise": base_noise,
                    "forced_steps": args.steps
                }
                
                if is_entropy:
                    logits_single, _ = _forward_logits(x, feature_extractor, concat_clf,
                                                    args, diff_fixed, h, w, False, K=1)
                    prob_single = torch.softmax(logits_single, dim=1)
                
                if args.single_forward:
                    num_steps = 1
                elif args.use_mc_dropout:
                    num_steps = args.mc_n_steps
                elif args.use_diffusion_stochastic:
                    steps_to_use = args.steps
                    num_steps = args.num_noisy_x
                else:
                    num_steps = 1

                inner_bar = tqdm(
                    total=num_steps,
                    desc=f"Img {idx}",
                    position=1,
                    leave=False,
                    dynamic_ncols=True,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} Forward [{elapsed}<{remaining}, {rate_fmt}]",
                )
                
                if args.single_forward:
                    logits, probs = _forward_logits(x, feature_extractor, concat_clf, args, diff_fixed, h, w, use_bald, K=1)
                    _append_probs_as_samples(logit_samples, probs)
                    pred = logits.argmax(dim=1).view(h, w).cpu()
                    inner_bar.update(1)
                
                elif args.use_mc_dropout:
                    # BALD: Bayesian Active Learning by Disagreement
                    num_samples = args.mc_n_steps
                    steps_to_use = args.steps
                    num_timesteps = len(steps_to_use)
                    
                    mega_batch_x = x.repeat(num_samples * num_timesteps, 1, 1, 1)
                    mega_batch_noise = base_noise.repeat(num_samples * num_timesteps, 1, 1, 1)
                    mega_batch_steps = torch.tensor(steps_to_use, device=x.device).repeat(num_samples)

                    diff_inputs = {
                        "noise": mega_batch_noise,
                        "forced_steps": mega_batch_steps.tolist(),
                    }
                    _, probs = _forward_logits(mega_batch_x, feature_extractor, concat_clf, args, diff_inputs, h, w, use_bald, K=args.mc_n_steps)
                    inner_bar.update(num_steps)
                    _append_probs_as_samples(logit_samples, probs)
                    
                    pred_logits, _ = _forward_logits(x, feature_extractor, concat_clf, args, diff_fixed, h, w, False, K=1)
                    pred = pred_logits.argmax(dim=1).view(h, w).cpu()
                    
                elif args.use_diffusion_stochastic:
                    # DALD: Diffusion-based Active Learning by Disagreement
                    num_samples = args.num_noisy_x
                    steps_to_use = args.steps
                    num_timesteps = len(steps_to_use)
                    total_forwards = num_samples * num_timesteps

                    g = torch.Generator(device=x.device).manual_seed(args.seed)
                    base_noise_samples = torch.randn(num_samples, *x.shape[1:], device=x.device, generator=g)
                    
                    mega_batch_x = x.repeat(total_forwards, 1, 1, 1)
                    mega_batch_noise = base_noise_samples.repeat_interleave(num_timesteps, dim=0)
                    mega_batch_steps = torch.tensor(steps_to_use, device=x.device).repeat(num_samples)

                    diff_inputs = {
                        "noise": mega_batch_noise,
                        "forced_steps": mega_batch_steps.tolist(),
                    }

                    inner_bar.total = total_forwards
                    inner_bar.refresh()

                    _, probs = _forward_logits(mega_batch_x, feature_extractor, concat_clf, args, diff_inputs, h, w, use_bald, K=args.num_noisy_x)

                    inner_bar.update(total_forwards)
                    _append_probs_as_samples(logit_samples, probs)
                    
                    pred_logits, _ = _forward_logits(x, feature_extractor, concat_clf, args, diff_fixed, h, w, False, K=1)
                    pred = pred_logits.argmax(dim=1).view(h, w).cpu()

                inner_bar.close()
                all_samples = x.new_empty(1, len(logit_samples), args.n_classes, h, w)
                for k, s in enumerate(logit_samples):
                    all_samples[0, k] = s.squeeze(0)
                
                if is_entropy:
                    uc_map = sampler(
                        {"entropy_probs": prob_single, "bald_logits": all_samples},
                        img_tag=img_tag,
                    )
                else:
                    sample_in = all_samples if use_bald else all_samples.squeeze(1)
                    uc_map = sampler(sample_in, img_tag=img_tag)
                
                uc_maps.append(uc_map.squeeze(0).cpu().numpy())
                pred_maps.append(pred)
            
    return uc_maps, pred_maps
