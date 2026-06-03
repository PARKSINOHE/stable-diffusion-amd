#!/usr/bin/env python3
"""Stable Diffusion on AMD ROCm GPUs"""
import torch, argparse, time
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def load_model(model_id="stabilityai/stable-diffusion-2-1", device="cuda"):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe

def generate(pipe, prompt, negative="", steps=25, width=512, height=512, seed=None):
    generator = torch.Generator(device=pipe.device).manual_seed(seed) if seed else None
    t0 = time.perf_counter()
    image = pipe(prompt, negative_prompt=negative, num_inference_steps=steps,
                 width=width, height=height, generator=generator).images[0]
    dt = time.perf_counter() - t0
    print(f"Generated in {dt:.2f}s")
    return image

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative", default="blurry, low quality")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--model", default="stabilityai/stable-diffusion-2-1")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output", default="output.png")
    args = p.parse_args()
    
    pipe = load_model(args.model)
    img = generate(pipe, args.prompt, args.negative, args.steps, args.width, args.height, args.seed)
    img.save(args.output)
    print(f"Saved to {args.output}")
