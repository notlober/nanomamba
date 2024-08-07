from dataclasses import dataclass
from typing import Optional
import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel as OriginalMambaLMHeadModel

@dataclass
class ModelArgs: 
    d_model: int = 768
    n_layer: int = 12
    vocab_size: int = 50304
    d_intermediate: int = 2048
    ssm_cfg: Optional[dict] = None
    attn_layer_idx: Optional[list] = None
    attn_cfg: Optional[dict] = None
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 16
    block_size: int = 1024
    tie_embeddings: bool = True

class MambaLMHeadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_args = ModelArgs(
            d_model=config.d_model,
            n_layer=config.n_layer,
            vocab_size=config.vocab_size,
            d_intermediate=config.d_intermediate,
            ssm_cfg=config.ssm_cfg,
            attn_layer_idx=config.attn_layer_idx,
            attn_cfg=config.attn_cfg,
            rms_norm=config.rms_norm,
            residual_in_fp32=config.residual_in_fp32,
            fused_add_norm=config.fused_add_norm,
            pad_vocab_size_multiple=config.pad_vocab_size_multiple,
            block_size=config.block_size,
            tie_embeddings=config.tie_embeddings
        )
        self.model = OriginalMambaLMHeadModel(model_args)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the Mamba model itself
        outputs = self.model(idx)
        logits = outputs.logits
        
        loss = None
        if targets is not None:
            logits_view = logits.view(-1, logits.size(-1))
            targets_view = targets.view(-1)
            
            # Calculate cross-entropy loss
            loss = F.cross_entropy(logits_view, targets_view, ignore_index=-1)
            
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        return cls(MambaConfig.from_pretrained(model_type))

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        print(f"Configured learning rate: {learning_rate}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def crop_block_size(self, block_size):
        # Mamba doesn't use positional embeddings, so we don't need to crop them
        # But we'll update the config
        self.config.block_size = block_size
        print(f"Cropped block size to {block_size}")