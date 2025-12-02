import math
import numpy as np
import torch
from tqdm import trange, tqdm

class GlobalHerding:
    def __init__(self, kernel_fn=None, sigma="auto", batch_size=1024,
                 device="cuda", dtype=None, seed=None):
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.kernel_fn = kernel_fn or RBFKernel(device=self.device)
        self.sigma = sigma
        self.batch_size = batch_size
        self.dtype = dtype

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def split_indices(self, idx_tensor, max_chunk_size):
        n = idx_tensor.numel()
        num_chunks = math.ceil(n / max_chunk_size)
        
        if num_chunks <= 1:
            return [idx_tensor]
            
        chunk_size = math.ceil(n / num_chunks)
        
        chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, n)
            chunks.append(idx_tensor[start:end])
            
        return chunks

    def estimate_split_size(self):
        mem_frac = 0.85  # Use 85% of free memory, leave 15% for other processes
        safety_overhead = 2.0  # Safety factor for kernel matrix memory overhead
        margin = 0.9  # Additional safety margin

        free_bytes, _ = torch.cuda.mem_get_info()
        usable = int(free_bytes * mem_frac)
        bytes_per_elem = torch.tensor([], dtype=self.dtype or torch.float32).element_size()
        
        n_max = int(math.sqrt(usable / (bytes_per_elem * safety_overhead)))
        n_max = int(n_max * margin)
        return max(4096, min(n_max, 50000))

    def compute_kernel_safe(self, X):
        torch.cuda.empty_cache()
        try:
            return self.kernel_fn.compute_kernel(
                X, X, h=self.sigma, batch_size=self.batch_size
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return None
            raise

    def herd_chunk(self, Xs, cur_max, budget, desc):
        if budget <= 0 or Xs.shape[0] == 0:
            return torch.empty(0, dtype=torch.long, device=Xs.device)

        K = self.compute_kernel_safe(Xs)
        if K is None:
            half = Xs.shape[0] // 2
            if half == 0:
                return torch.arange(0, 1, device=Xs.device, dtype=torch.long)

            b1, b2 = budget // 2, budget - budget // 2
            X1, X2 = Xs[:half], Xs[half:]
            cm1, cm2 = cur_max[:half], cur_max[half:]

            sel1 = self.herd_chunk(X1, cm1, b1, desc)
            if sel1.numel() > 0:
                cross12 = self.kernel_fn.compute_kernel(
                    X2, X1[sel1], h=self.sigma, batch_size=self.batch_size
                )
                cm2 = torch.maximum(cm2, cross12.max(1).values)

            sel2 = self.herd_chunk(X2, cm2, b2, desc)
            return torch.cat([sel1, sel2 + half])

        cur_max = cur_max.clone()
        mask = torch.zeros(Xs.shape[0], dtype=torch.bool, device=Xs.device)

        t = trange(budget, desc=desc, leave=True)
        for _ in t:
            gain = torch.clamp(K - cur_max.unsqueeze(1), min=0).mean(0)
            gain[mask] = -1e4
            best = torch.argmax(gain)
            mask[best] = True
            cur_max = torch.maximum(cur_max, K[best])

        return torch.where(mask)[0]

    def greedy_select(self, feats, budget, num_prev=0, allow_split=True):
        X = feats.to(self.device, dtype=self.dtype, non_blocking=True)
        M, _ = X.shape

        if budget > (M - num_prev):
            budget = M - num_prev
        if budget <= 0:
            return np.empty((0,), dtype=np.int64)

        self.sigma = auto_sigma(X.float(), sample_ratio=0.01) if self.sigma == "auto" else self.sigma

        idx_all = torch.arange(M, device=self.device)
        idx_prev = idx_all[:num_prev] if num_prev > 0 else None
        idx_new = idx_all[num_prev:]
        idx_new = idx_new[torch.randperm(idx_new.numel(), device=self.device)]

        safe_chunk_size = self.estimate_split_size()
        total_new = idx_new.numel()
        
        use_split_strategy = False
        if allow_split and (total_new > safe_chunk_size):
            use_split_strategy = True
            print(f"[Global Herding] Estimator predicted OOM. Switching to SPLIT mode directly.")

        selected = None
        
        if not use_split_strategy:
            print(f"[Global Herding] Processing {total_new} candidates")
            try:
                selected = self._run_herding(X, [idx_new], budget, num_prev, idx_prev, total_new)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and allow_split:
                    print(f"[Global Herding] OOM! Clearing cache and falling back to SPLIT mode...")
                    torch.cuda.empty_cache()
                    use_split_strategy = True
                else:
                    raise e

        if use_split_strategy:
            max_split = max(safe_chunk_size, 8192)
            chunks = self.split_indices(idx_new, max_split)
            
            chunk_sizes = [ch.numel() for ch in chunks]
            if len(chunks) == 1:
                print(f"[Global Herding] Running Split Mode ({chunk_sizes[0]} candidates)")
            else:
                print(f"[Global Herding] Running Split Mode: {len(chunks)} splits (sizes: {chunk_sizes}, max_split_size={max_split})")
            
            selected = self._run_herding(X, chunks, budget, num_prev, idx_prev, total_new)

        return selected

    def _run_herding(self, X, chunks, budget, num_prev, idx_prev, total_new):
        num_chunks = len(chunks)
        if num_chunks > 1:
            print(f"[_run_herding] Processing {num_chunks} chunks with total budget={budget}")
        
        selected = []
        remaining = budget
        
        for ci, ch_idx in enumerate(chunks, 1):
            n = ch_idx.numel()
            if n == 0 or remaining <= 0:
                continue

            if len(chunks) == 1:
                b = remaining
            else:
                b = remaining if ci == len(chunks) else min(
                    remaining, int(round(budget * (n / total_new)))
                )
            
            if num_chunks > 1:
                print(f"  > Chunk {ci}/{num_chunks}: size={n}, local_budget={b}")

            Xs = X[ch_idx]

            if num_prev > 0:
                prev_feats = X[idx_prev]
                max_prev = 5000 
                if prev_feats.shape[0] > max_prev:
                    idx_sample = torch.randperm(prev_feats.shape[0], device=prev_feats.device)[:max_prev]
                    prev_feats = prev_feats[idx_sample]

                cross_prev = self.kernel_fn.compute_kernel(
                    Xs, prev_feats, h=self.sigma, batch_size=self.batch_size
                )
                cur_max = cross_prev.max(1).values
            else:
                cur_max = torch.zeros(n, device=self.device, dtype=self.dtype)

            if len(selected) > 0:
                sel_feats = X[torch.cat(selected)]
                max_sel = 5000
                if sel_feats.shape[0] > max_sel:
                    idx_sample = torch.randperm(sel_feats.shape[0], device=sel_feats.device)[:max_sel]
                    sel_feats = sel_feats[idx_sample]

                cross_sel = self.kernel_fn.compute_kernel(
                    Xs, sel_feats, h=self.sigma, batch_size=self.batch_size
                )
                cur_max = torch.maximum(cur_max, cross_sel.max(1).values)

            desc = "Global Herding" if len(chunks) == 1 else f"Split Herding {ci}/{len(chunks)}"
            sel_local = self.herd_chunk(Xs, cur_max, b, desc)

            if sel_local.numel() > 0:
                selected.append(ch_idx[sel_local])

            remaining -= b

        if not selected:
            return np.empty((0,), dtype=np.int64)
            
        final_selection = torch.cat(selected, dim=0)
        if final_selection.numel() > budget:
            final_selection = final_selection[:budget]
            
        return final_selection.detach().cpu().numpy()

def sample_local_candidates(args, label):
    rng = np.random.default_rng(args.seed)
    valid_pixels = np.where(label.flatten() != args.ignore_index)[0]
    if len(valid_pixels) >= args.local_candidate:
        chosen_pixels = rng.choice(valid_pixels, args.local_candidate, replace=False)
    else:
        chosen_pixels = valid_pixels
    candidate_mask_flat = np.zeros(label.size, dtype=bool)
    candidate_mask_flat[chosen_pixels] = True
    candidate_mask = candidate_mask_flat.reshape(label.shape)
    return candidate_mask

def compute_norm(x1: torch.Tensor, x2: torch.Tensor, device='cuda', batch_size=2048) -> torch.Tensor:
    x1 = x1.unsqueeze(0).to(device)
    x2 = x2.unsqueeze(0).to(device)
    dist_list = []
    m = x2.shape[1]
    num_batches = (m // batch_size) + int((m % batch_size) > 0)

    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        x2_subset = x2[:, start:end]

        dist_chunk = torch.cdist(x1, x2_subset, p=2.0)
        dist_list.append(dist_chunk)

    dist_matrix = torch.cat(dist_list, dim=-1).squeeze(0)
    return dist_matrix

class RBFKernel:
    def __init__(self, device='cuda'):
        self.device = device
    def compute_kernel(self, x1: torch.Tensor, x2: torch.Tensor, h=1.0, batch_size=2048) -> torch.Tensor:
        norm = compute_norm(x1, x2, device=self.device, batch_size=batch_size)
        k = torch.exp(- (norm / h) ** 2)
        return k

class NegNormKernel:
    def __init__(self, device):
        self.device = device
    def compute_kernel(self, x1, x2, h, batch_size=2048):
        dist_matrix = compute_norm(x1, x2, self.device, batch_size=batch_size)
        return -dist_matrix

class TopHatKernel:
    def __init__(self, device):
        self.device = device
    def compute_kernel(self, x1, x2, h, batch_size=2048):
        x1 = x1.unsqueeze(0).to(self.device)
        x2 = x2.unsqueeze(0).to(self.device)
        dist_matrix_list = []
        m = x2.shape[1]
        num_batches = m // batch_size + int(m % batch_size > 0)
        for i in range(num_batches):
            x2_subset = x2[:, i * batch_size: (i + 1) * batch_size]
            dist = torch.cdist(x1, x2_subset)
            dist_matrix_list.append(dist.cpu())
        dist_matrix = torch.cat(dist_matrix_list, dim=-1).squeeze(0)
        k = (dist_matrix < h)
        return k

class StudentTKernel:
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h=1.0, batch_size=2048, beta=0.5):
        norms = compute_norm(x1, x2, self.device, batch_size=batch_size)
        k = (1 + ((norms / h) ** 2) / beta) ** (-(beta+1)/2)
        return k

class LaplaceKernel:
    def __init__(self, device):
        self.device = device
    def compute_kernel(self, x1, x2, h=1.0, batch_size=2048, beta=1):
        norms = compute_norm(x1, x2, self.device, batch_size=batch_size)
        k = torch.exp(- (norms ** beta) / h)
        return k

class CauchyKernel:
    def __init__(self, device):
        self.device = device
    def compute_kernel(self, x1, x2, h=1.0, batch_size=2048):
        norms = compute_norm(x1, x2, self.device, batch_size=batch_size)
        k = 1 / (1 + norms ** 2)
        return k

class RationalQuadKernel:
    def __init__(self, device):
        self.device = device
    def compute_kernel(self, x1, x2, h=1.0, batch_size=2048, alpha=1.0):
        norms = compute_norm(x1, x2, self.device, batch_size=batch_size)
        k = (1 + norms ** 2 / (2 * alpha)) ** (-alpha)
        return k

def auto_sigma(X, sample_ratio=0.01, scale_factor=2.0):
    N = X.shape[0]
    sub_n = int(N * sample_ratio)
    sub_n = max(sub_n, 10)

    if sub_n < N:
        idx = np.random.choice(N, size=sub_n, replace=False)
        sub_feats = X[idx]
    else:
        sub_feats = X
        sub_n = sub_feats.shape[0]

    sub_feats = sub_feats.float()
    dist_matrix = torch.cdist(sub_feats, sub_feats, p=2)
    tri_idx = torch.triu_indices(sub_n, sub_n, offset=1)
    dists = dist_matrix[tri_idx[0], tri_idx[1]]
    dists = dists[dists > 0]

    if len(dists) == 0:
        sigma = 1.0
    else:
        sigma = torch.median(dists).item()

    sigma = max(sigma, 1e-6) * scale_factor
    return sigma


def local_herding(
    candidate_features: torch.Tensor,
    budget: int,
    kernel_fn=None,
    sigma='auto',
    percentile=None,
    batch_size=2048,
    device='cuda',
    prev_rep_features: torch.Tensor = None,
    disable_tqdm: bool = False,
    progress_desc: str = "Local Herding"
):
    if kernel_fn is None:
        kernel_fn = RBFKernel(device)
    candidate_features = candidate_features.to(device)
    N, d = candidate_features.shape
    if sigma == 'auto':
        sigma = auto_sigma(candidate_features, sample_ratio=0.01)
    K = kernel_fn.compute_kernel(candidate_features, candidate_features, h=sigma, batch_size=batch_size)
    K = K.to(device)
    
    if prev_rep_features is not None and prev_rep_features.shape[0] > 0:
        prev_rep_features = prev_rep_features.to(device)
        K_prev = kernel_fn.compute_kernel(candidate_features, prev_rep_features, h=sigma, batch_size=batch_size)
        current_max = K_prev.max(dim=1)[0]
        selected = []
        selection_order = []
        init_idx = None
        init_cov = current_max.mean().item()
    else:
        avg_K = K.mean(dim=1)
        init_idx = torch.argmax(avg_K).item()
        selected = [init_idx]
        selection_order = [init_idx]
        current_max = K[init_idx].clone()
        init_cov = current_max.mean().item()
    
    coverage = current_max.mean()
    is_selected = torch.zeros(N, dtype=torch.bool, device=device)
    for idx in selected:
        is_selected[idx] = True

    t = tqdm(range(len(selected), budget), desc=progress_desc, leave=False, disable=disable_tqdm)
    for _ in t:
        max_mat = torch.maximum(current_max.unsqueeze(1), K)
        coverage_j = max_mat.mean(dim=0)
        coverage_gain = coverage_j - coverage
        coverage_gain[is_selected] = -1e4
        best_idx = torch.argmax(coverage_gain).item()
        best_gain = coverage_gain[best_idx].item()
        selected.append(best_idx)
        selection_order.append(best_idx)
        is_selected[best_idx] = True
        current_max = torch.maximum(current_max, K[best_idx])
        coverage = current_max.mean()
        t.set_postfix({
            "Best Gain": f"{best_gain:.4f}",
            "Selected Idx": best_idx,
            "Coverage": f"{coverage:.4f}"
        })
    
    return np.array(selected, dtype=np.int64), np.array(selection_order, dtype=np.int64), init_idx, init_cov

