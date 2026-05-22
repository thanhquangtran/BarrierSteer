#!/usr/bin/env python3
import argparse

from steer.reft_r1.trainer import train_reft_r1


def parse_args():
    p = argparse.ArgumentParser(description="Train a rank-1 ReFT vector for HarmBench")
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--target_layer", type=int, required=True)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--l1_coeff", type=float, default=0.0)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt = train_reft_r1(**vars(args))
    print(f"Saved ReFT-r1 vector to {ckpt}")
