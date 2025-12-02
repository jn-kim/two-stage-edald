import torch
import torch.nn as nn
import torch.nn.functional as F
import guided_diffusion.dist_util as dist_util
from guided_diffusion.script_util import create_model_and_diffusion
import inspect
from typing import List
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def prepare_concat_features(batch_img, batch_label, feature_extractor, args, noise=None):
    processed_labels = batch_label.clone().flatten()
    batch_size = len(batch_img)
    batch_noise = None if noise is None else noise.expand(batch_size, *noise.shape[1:])
    
    features = feature_extractor(batch_img, noise=batch_noise)
    collected_features, sumC = collect_batch_features(args, features)
    
    B, _, H, W = collected_features.shape
    all_concat_feature = collected_features.permute(0, 2, 3, 1).reshape(B * H * W, sumC)
    
    del features, batch_noise
    torch.cuda.empty_cache()
    
    return all_concat_feature, processed_labels

def collect_batch_features(args, activations: List[torch.Tensor]):
    assert all(isinstance(a, torch.Tensor) for a in activations)
    target_size = (args.img_size, args.img_size)
    
    batch_size_combined = activations[0].shape[0]
    num_timesteps = len(args.steps) if batch_size_combined % len(args.steps) == 0 else 1
    
    original_batch_size = batch_size_combined // num_timesteps
    
    timestep_features_list = [[] for _ in range(num_timesteps)]
    
    for tensor in activations:
        reshaped_tensor = tensor.view(original_batch_size, num_timesteps, *tensor.shape[1:])
        split_tensors = list(torch.split(reshaped_tensor, 1, dim=1))
        
        for t in range(num_timesteps):
            timestep_features_list[t].append(split_tensors[t].squeeze(dim=1))
    
    concatenated_timesteps = []
    total_channel_sum = 0
    for features_for_one_timestep in timestep_features_list:
        resized_list = []
        current_sum_channels = 0
        for tensor in features_for_one_timestep:
            resized_tensor = F.interpolate(
                tensor, size=target_size, mode='bilinear', align_corners=False
            )
            resized_list.append(resized_tensor)
            current_sum_channels += tensor.shape[1]
        
        concatenated_one_timestep = torch.cat(resized_list, dim=1)
        concatenated_timesteps.append(concatenated_one_timestep)
        
        if total_channel_sum == 0:
            total_channel_sum = current_sum_channels

    concatenated_features = torch.cat(concatenated_timesteps, dim=1)
    
    B, D, H, W = concatenated_features.shape
    sumC = D
    
    reshaped = concatenated_features.permute(0, 2, 3, 1)  
    reshaped = reshaped.reshape(B * H * W, D)  
    reshaped = F.normalize(reshaped, dim=1)  
    normalized_features = reshaped.reshape(B, H, W, D).permute(0, 3, 1, 2)
    
    del resized_list, resized_tensor
    torch.cuda.empty_cache()
    
    return normalized_features, sumC

def collect_features(self, activations: List[torch.Tensor], sample_idx=0):
    assert all(isinstance(acts, torch.Tensor) for acts in activations)
    size = (self.img_size, self.img_size)
    resized_activations = []

    for idx, feats in enumerate(activations):
        feats_selected = feats[sample_idx][None]
        feats_resized = torch.nn.functional.interpolate(
            feats_selected, size=size, mode="bilinear"
        )
        resized_activations.append(feats_resized[0])
        del feats, feats_selected, feats_resized
        torch.cuda.empty_cache()

    concatenated_features = torch.cat(resized_activations, dim=0)

    D, H, W = concatenated_features.shape
    reshaped = concatenated_features.view(D, -1).permute(1, 0)
    reshaped = torch.nn.functional.normalize(reshaped, dim=1)
    concatenated_features = reshaped.permute(1, 0).view(D, H, W)

    del resized_activations
    torch.cuda.empty_cache()

    sumC = concatenated_features.shape[0]
    return concatenated_features, sumC

class FeatureExtractorDDPM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._load_pretrained_model(args)
        
        self.default_steps = args.steps
        self.default_blocks = args.blocks

        for param in self.model.parameters():
            param.requires_grad = False

    def _load_pretrained_model(self, args):
        argnames = inspect.getfullargspec(create_model_and_diffusion)[0]
        model_args = {name: getattr(args, name) for name in argnames if hasattr(args, name)}
        self.model, self.diffusion = create_model_and_diffusion(**model_args)
        
        ckpt = torch.load(args.model_path)
        self.model.load_state_dict(ckpt)
        print(f"Pretrained model loaded from {args.model_path}")
            
        self.model.to(dist_util.dev())
        if getattr(args, "use_fp16", False):
            self.model.convert_to_fp16()
            
    def forward(self, x, noise=None, forced_steps=None):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        steps_to_use = self.default_steps if forced_steps is None else forced_steps
        if not torch.is_tensor(steps_to_use):
            steps_to_use = torch.as_tensor(steps_to_use, device=x.device)
        else:
            steps_to_use = steps_to_use.to(x.device)
        steps_to_use = steps_to_use.view(-1)

        B = x.size(0)
        M = steps_to_use.numel()

        if forced_steps is not None and M == B:
            t_batch = steps_to_use
            noisy_x_batch = self.diffusion.q_sample(x, t_batch, noise=noise).to(x.dtype)
            final_activations = self.model(noisy_x_batch, self.diffusion._scale_timesteps(t_batch), block_idx=self.default_blocks)
            return final_activations
            
        if forced_steps is not None and B == 1 and M > 1:
            activations = []
            for t in steps_to_use.tolist():
                t_tensor = torch.tensor([t], device=x.device)
                noisy_x = self.diffusion.q_sample(x, t_tensor, noise=noise).to(x.dtype)
                feats = self.model(noisy_x, self.diffusion._scale_timesteps(t_tensor), block_idx=self.default_blocks)
                activations.extend(feats)

            return activations
        
        elif forced_steps is None:
            x_rep = x.repeat_interleave(M, dim=0)
            t_rep = steps_to_use.repeat(B)

            if noise is not None:
                if noise.size(0) == B:
                    z = noise.repeat_interleave(M, dim=0)
                elif noise.size(0) == x_rep.size(0):
                    z = noise
            else:
                z = None

            noisy_x_batch = self.diffusion.q_sample(x_rep, t_rep, noise=z).to(x_rep.dtype)
            final_activations = self.model(noisy_x_batch, self.diffusion._scale_timesteps(t_rep), block_idx=self.default_blocks)
            return final_activations