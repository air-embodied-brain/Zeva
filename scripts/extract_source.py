import argparse
import logging
import os
import dataclasses
from pathlib import Path
from typing import Dict, Any

import torch
import tqdm
import numpy as np
import tyro
import jax 


import openpi.training.config as _config
import openpi.models.pi0_config as _pi0_config
import openpi.models_pytorch.pi0_pytorch as _pi0_pytorch
import openpi.training.data_loader as _data_loader
import openpi.models.model as _model
import safetensors.torch


logging.basicConfig(level=logging.INFO)

@dataclasses.dataclass
class Args:
    config_name: str
    ckpt_path: str
    save_path: str = "vlm_source_features.pt"
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Optional limit for smoke tests. Leave unset for the full memory bank.
    max_episodes: int | None = None

def load_model(config: _config.TrainConfig, ckpt_path: str, device: str):
    logging.info(f"Loading model architecture: {config.model.model_type}")
    
    if not isinstance(config.model, _pi0_config.Pi0Config):
        model_cfg = _pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        # Task-schema extraction only needs the VLM. Disable Zeva modules so
        # online causal checkpoints are not required during offline extraction.
        model_cfg = dataclasses.replace(
            config.model,
            dtype=config.pytorch_training_precision,
            use_zeva=False,
            use_action_prior=False,
            is_training=True,
            schema_retrieval_ckpt=None,
            schema_memory_path=None,
            causal_encoder_ckpt=None,
            causal_adapter_ckpt=None,
        )

    model = _pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    model.eval()
    
    ckpt_path = Path(ckpt_path)
    if ckpt_path.is_dir():
        ckpt_file = ckpt_path / "model.safetensors"
    else:
        ckpt_file = ckpt_path

    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_file}")

    logging.info(f"Loading weights from {ckpt_file}...")
    missing, unexpected = safetensors.torch.load_model(model, ckpt_file, strict=False)
    if missing:
        raise RuntimeError(f"Missing VLM checkpoint tensors: {sorted(missing)}")
    if unexpected:
        logging.info("Ignored %d Zeva-only checkpoint tensors.", len(unexpected))
    
    # --- DEBUG OUTPUT ---
    vlm_hidden_dim = model.paligemma_with_expert.paligemma.config.text_config.hidden_size
    logging.info(f"DEBUG: Model Type: PI05={model_cfg.pi05}, VLM Hidden Dim: {vlm_hidden_dim}")
    logging.info(f"DEBUG: Using precision: {model_cfg.dtype} on device {device}")
    # --------------------
    
    logging.info("Model loaded successfully.")
    
    return model

def get_dataset_root(dataset):
    lerobot_ds = dataset
    while hasattr(lerobot_ds, "_dataset"):
        lerobot_ds = lerobot_ds._dataset
    return lerobot_ds

def main(args: Args):
    logging.info(f"Loading configuration: {args.config_name}")
    train_config = _config.get_config(args.config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    
    logging.info("Building dataset pipeline...")
    base_dataset = _data_loader.create_torch_dataset(
        data_config, 
        action_horizon=train_config.model.action_horizon, 
        model_config=train_config.model
    )
    dataset = _data_loader.transform_dataset(
        base_dataset, 
        data_config, 
        skip_norm_stats=False 
    )
    
    lerobot_ds = get_dataset_root(dataset)
    if not hasattr(lerobot_ds, "episode_data_index"):
        raise ValueError("Could not find episode_data_index in dataset.")
        
    episode_starts = lerobot_ds.episode_data_index["from"].tolist()
    if args.max_episodes is not None:
        episode_starts = episode_starts[:args.max_episodes]
    total_episodes = len(episode_starts)
    logging.info(f"Found {total_episodes} episodes in dataset.")

    model = load_model(train_config, args.ckpt_path, args.device)
    all_features = []
    metadata = {
        "episode_indices": [], 
        "task_indices": []
    }
    
    collate_fn = _data_loader._collate_fn
    logging.info(f"Starting extraction with batch size {args.batch_size}...")
    
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, total_episodes, args.batch_size)):
            batch_frame_indices = episode_starts[i : i + args.batch_size]
            
            raw_batch = [dataset[idx] for idx in batch_frame_indices]
            
            try:
                task_indices_raw = [lerobot_ds[idx]["task_index"] for idx in batch_frame_indices]
            except KeyError:
                logging.error("CRITICAL: 'task_index' not found in raw dataset sample. Cannot save metadata.")
                raise
                
            task_indices_batch = torch.as_tensor(task_indices_raw, dtype=torch.long)
            
            batch_data = collate_fn(raw_batch)

            def to_device_and_squeeze(x):
                t = torch.as_tensor(x).to(args.device)
    
                if t.ndim >= 2 and t.shape[1] == train_config.model.action_horizon:
                    t = t[:, 0]
                                
                if t.dtype in [torch.uint8, torch.int64, torch.int32]:
                    if t.dtype == torch.uint8:
                         t = t.float() 
                    else:
                         t = t.float() 
                return t

            images_dict = {}
            image_masks_dict = {}
            
            for key, val in batch_data["image"].items():
                img_tensor = to_device_and_squeeze(val)
                
                if img_tensor.ndim == 4 and img_tensor.shape[-1] == 3: 
                    img_tensor = img_tensor.permute(0, 3, 1, 2).contiguous() 

                images_dict[key] = img_tensor
                
                if "image_mask" in batch_data and key in batch_data["image_mask"]:
                    mask_val = batch_data["image_mask"][key]
                    mask_tensor = torch.as_tensor(mask_val).to(args.device, dtype=torch.bool)
                    if mask_tensor.ndim > 1 and mask_tensor.shape[1] == train_config.model.action_horizon:
                        mask_tensor = mask_tensor[:, 0]
                    if mask_tensor.ndim == 0:
                        mask_tensor = mask_tensor.expand(img_tensor.shape[0])
                    image_masks_dict[key] = mask_tensor
                else:
                    image_masks_dict[key] = torch.ones(img_tensor.shape[0], dtype=torch.bool, device=args.device)

            state = to_device_and_squeeze(batch_data["state"])
            prompt = torch.as_tensor(batch_data["tokenized_prompt"]).to(args.device)
            if prompt.ndim == 3 and prompt.shape[1] == train_config.model.action_horizon:
                prompt = prompt[:, 0]
                
            prompt_mask = (prompt != 0)

            obs = _model.Observation(
                images=images_dict,
                image_masks=image_masks_dict,
                state=state,
                tokenized_prompt=prompt.long(), 
                tokenized_prompt_mask=prompt_mask.bool(),
                token_ar_mask=None,
                token_loss_mask=None
            )
            
            feats = model.extract_vlm_features(obs)            
            feats = torch.nn.functional.normalize(feats, dim=-1)
            all_features.append(feats.cpu())
            
            current_indices = list(range(i, i + len(batch_frame_indices)))
            metadata["episode_indices"].extend(current_indices)
            metadata["task_indices"].extend(task_indices_batch.flatten().cpu().tolist())

    
    final_features = torch.cat(all_features, dim=0)
    
    logging.info(f"DEBUG: Final extracted features shape: {final_features.shape}")
    
    save_payload = {
        "features": final_features,
        "episode_indices": metadata["episode_indices"],
        "task_indices": metadata["task_indices"]
    }
    
    logging.info(f"Extraction complete. Features shape: {final_features.shape}")
    logging.info(f"Saving to {args.save_path}...")
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_payload, args.save_path)

if __name__ == "__main__":
    main(tyro.cli(Args))
