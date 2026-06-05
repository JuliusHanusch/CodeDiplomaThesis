import sys  
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir.resolve()))  
sys.path.append(str((root_dir/"src").resolve()))  
sys.path.append(str((root_dir / "chronos_pkg/src").resolve()))

import torch
from transformers import T5Config
from chronos_pkg.src.chronos.chronos_bolt import ChronosBoltModelForForecasting

def build_toy_config():
    return T5Config(
        d_model=64,
        d_ff=128,
        num_layers=2,
        num_decoder_layers=2,
        vocab_size=1,
        decoder_start_token_id=0,
        dense_act_fn="relu",
        dropout_rate=0.0,
        initializer_factor=1.0,
        chronos_config=dict(
            context_length=32,
            prediction_length=8,
            input_patch_size=4,
            input_patch_stride=4,
            quantiles=[0.1, 0.5, 0.9],
            use_reg_token=False,
            individual_dropout=0.05, # TODO add to search space
            mask_prob=0.15,          # TODO add to search space
            perturbation_prob=0.05,  # TODO add to search space
            perturbation_strength=0.1, # TODO add to search space
            unchanged_patch_prob=0.05, # TODO add to search space
            patch_perturbation_prob=0.1, # TODO add to search space
            patch_perturbation_scale_strength=0.1, # TODO add to search space
            patch_perturbation_noise_strength=0.1, # TODO add to search space
            MLMmodel="RoBERTa", # TODO add to search space ("RoBERTa" or "T5")
        ),
        architectures=["ChronosBoltModelForForecasting"],
    )

def build_toy_batch():
    # batch_size=2, sequence_length=32
    ctx = torch.randn(2, 32)
    mask = torch.ones_like(ctx)
    tgt = torch.randn(2, 8)
    tgt_mask = torch.ones_like(tgt)
    return ctx, mask, tgt, tgt_mask

if __name__ == "__main__":
    cfg = build_toy_config()
    model = ChronosBoltModelForForecasting(cfg)

    ctx, mask, tgt, tgt_mask = build_toy_batch()

    out = model(context=ctx, target=ctx)
    print("Loss:", out.loss.item() if out.loss is not None else None)
    print("Quantile preds shape:", out.quantile_preds.shape)
