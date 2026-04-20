import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        in_f = original_linear.in_features
        out_f = original_linear.out_features
        self.scale = alpha / rank

        # lora_ prefix is what unet_trainable_list will match on
        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B = 0 means zero delta at init → pretrained behavior preserved

    def forward(self, x):
        return self.original_linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


def _should_lora(param_name: str, target_temporal: bool = True) -> bool:
    """
    Decide whether a given parameter path should get LoRA.
    Works directly on the actual parameter name strings from the checkpoint.
    """
    # Must be a weight matrix (not bias, norm, or embedding)
    if not param_name.endswith('.weight'):
        return False

    # Skip spatial self-attention entirely (attn1 inside .1 blocks)
    # Pattern: blocks.X.1.transformer_blocks.Y.attn1
    # We detect .1. followed eventually by attn1 with NO .2. in between
    # Simpler: if attn1 and NOT inside a .2. temporal block
    is_attn1 = '.attn1.' in param_name
    is_attn2 = '.attn2.' in param_name
    is_temporal_block = bool(
        # input/output blocks: .X.2. pattern
        ('.1.transformer' not in param_name.split('.attn')[0].split('blocks.')[-1]
         if 'input_blocks' in param_name or 'output_blocks' in param_name else False)
    )

    # Cleaner approach: just check the second-to-last block index
    # input_blocks.X.IDX.transformer_blocks → IDX=1 is spatial, IDX=2 is temporal
    parts = param_name.split('.')
    try:
        # find 'transformer_blocks' position and look 2 back for the block index
        tb_idx = parts.index('transformer_blocks')
        block_subindex = int(parts[tb_idx - 1])  # 1 = spatial, 2 = temporal
    except (ValueError, IndexError):
        block_subindex = None

    # middle_block has indices 1 and 2 directly (middle_block.1 and middle_block.2)
    is_middle = 'middle_block' in param_name

    # Rule 1: attn2 in spatial blocks (.1) → ALWAYS include (cross-attention + _ip/_tp)
    if is_attn2 and block_subindex == 1:
        return True

    # Rule 2: temporal blocks (.2) → include if target_temporal=True
    if block_subindex == 2 and target_temporal:
        # both attn1 and attn2 in temporal blocks are useful
        if is_attn1 or is_attn2:
            return True

    # Rule 3: middle_block cross-attention (both .1 and .2 slots)
    if is_middle and is_attn2:
        return True

    # Rule 4: attn1 in spatial blocks → skip (frozen, pretrained visual priors)
    # (implicitly handled by not matching above)

    return False


def _get_submodule(model, path: str):
    """Navigate to a submodule given a dot-separated path."""
    parts = path.split('.')
    mod = model
    for p in parts:
        if p.isdigit():
            mod = mod[int(p)]
        else:
            mod = getattr(mod, p)
    return mod


def _set_submodule_attr(model, path: str, new_module: nn.Module):
    """Set an attribute at the end of a dot-separated path."""
    parts = path.split('.')
    parent = _get_submodule(model, '.'.join(parts[:-1]))
    attr = parts[-1]
    if attr.isdigit():
        parent[int(attr)] = new_module
    else:
        setattr(parent, attr, new_module)


def inject_lora_into_unet(diffusion_model, rank: int = 16, alpha: float = 32.0,
                           target_temporal: bool = True):
    """
    Injects LoRA into the diffusion UNet using exact parameter name matching.
    Must be called AFTER checkpoint loading.
    
    diffusion_model: self.model.diffusion_model  (the actual UNet, not the wrapper)
    """
    injected = []
    skipped_bias = []

    # Collect all Linear layers that should get LoRA
    targets = {}
    for name, module in diffusion_model.named_modules():
        if isinstance(module, nn.Linear):
            # Construct the weight param name as it appears in checkpoint
            weight_name = f"diffusion_model.{name}.weight"
            if _should_lora(weight_name, target_temporal):
                targets[name] = module

    # Replace them
    for name, module in targets.items():
        lora_module = LoRALinear(module, rank=rank, alpha=alpha)
        _set_submodule_attr(diffusion_model, name, lora_module)
        injected.append(name)

    print(f"LoRA injected into {len(injected)} Linear layers")
    print(f"Example targets: {injected[:3]} ... {injected[-3:]}")
    return diffusion_model