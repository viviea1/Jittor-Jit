#!/usr/bin/env python3
"""Two-node Jittor JiT-B/16 training benchmark on real ImageNet data.

The model forward uses BF16 tensors.  Gradients, AdamW state, master weights,
and both EMA copies stay in FP32 so the numerical roles match the reference
Torch BF16-autocast run as closely as Jittor 1.3.11 permits.
"""

import argparse
import gc
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import jittor as jt
import numpy as np
from jittor import nn
from jittor.dataset import ImageFolder
from PIL import Image

from model_jit import JiT_B_16


def center_crop_arr(image, image_size):
    while min(image.size) >= 2 * image_size:
        image = image.resize(
            tuple(value // 2 for value in image.size), resample=Image.BOX
        )
    scale = image_size / min(image.size)
    image = image.resize(
        tuple(round(value * scale) for value in image.size),
        resample=Image.BICUBIC,
    )
    array = np.asarray(image)
    crop_y = (array.shape[0] - image_size) // 2
    crop_x = (array.shape[1] - image_size) // 2
    return array[
        crop_y : crop_y + image_size,
        crop_x : crop_x + image_size,
    ]


class ImageTransform:
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, image):
        array = center_crop_arr(image, self.image_size)
        if np.random.random() < 0.5:
            array = array[:, ::-1]
        return np.ascontiguousarray(array.transpose(2, 0, 1), dtype=np.uint8)


class TrainingDenoiser(nn.Module):
    def __init__(
        self,
        image_size=256,
        num_classes=1000,
        label_drop_prob=0.1,
        p_mean=-0.8,
        p_std=0.8,
        noise_scale=1.0,
        t_eps=0.05,
    ):
        super().__init__()
        self.net = JiT_B_16(
            input_size=image_size,
            in_channels=3,
            num_classes=num_classes,
        )
        self.num_classes = num_classes
        self.label_drop_prob = label_drop_prob
        self.p_mean = p_mean
        self.p_std = p_std
        self.noise_scale = noise_scale
        self.t_eps = t_eps

    def execute(self, images, labels):
        batch = int(images.shape[0])
        drop = jt.rand((batch,)) < self.label_drop_prob
        labels_dropped = jt.where(
            drop,
            jt.ones_like(labels) * self.num_classes,
            labels,
        )

        t = jt.sigmoid(
            jt.randn((batch,), dtype="float32") * self.p_std + self.p_mean
        ).reshape((batch, 1, 1, 1))
        noise = jt.randn(images.shape, dtype="float32") * self.noise_scale
        z = t * images + (1.0 - t) * noise
        denominator = jt.maximum(
            1.0 - t,
            jt.ones_like(t) * self.t_eps,
        )
        target_velocity = (images - z) / denominator

        prediction = self.net(z, t.flatten(), labels_dropped)
        predicted_velocity = (prediction.float32() - z) / denominator
        difference = target_velocity.float32() - predicted_velocity.float32()
        return (difference * difference).mean()


class MasterAdamW:
    """BF16 model parameters with FP32 master weights and optimizer state."""

    def __init__(
        self,
        named_parameters,
        lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        accumulation_steps=4,
        ema_decays=(0.9999, 0.9996),
        master_initial_values=None,
    ):
        trainable = [
            (name, parameter)
            for name, parameter in named_parameters
            if not parameter.is_stop_grad()
        ]
        self.names = [name for name, _ in trainable]
        self.params = [parameter for _, parameter in trainable]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.accumulation_steps = accumulation_steps
        self.ema_decays = ema_decays
        self.micro_step = 0
        self.optimizer_step = 0

        if master_initial_values is None:
            initial_values = [
                parameter.float32().copy().stop_grad()
                for parameter in self.params
            ]
        else:
            missing = [
                name for name in self.names if name not in master_initial_values
            ]
            if missing:
                raise KeyError(
                    "missing FP32 master initial values: " + ", ".join(missing)
                )
            initial_values = [
                master_initial_values[name].float32().copy().stop_grad()
                for name in self.names
            ]

        self.master = initial_values
        self.exp_avg = [
            jt.zeros(parameter.shape, dtype="float32").stop_grad()
            for parameter in self.params
        ]
        self.exp_avg_sq = [
            jt.zeros(parameter.shape, dtype="float32").stop_grad()
            for parameter in self.params
        ]
        self.grad_accum = [
            jt.zeros(parameter.shape, dtype="float32").stop_grad()
            for parameter in self.params
        ]
        self.ema1 = [value.copy().stop_grad() for value in initial_values]
        self.ema2 = [value.copy().stop_grad() for value in initial_values]
        jt.sync(
            self.master
            + self.exp_avg
            + self.exp_avg_sq
            + self.grad_accum
            + self.ema1
            + self.ema2
        )

    def backward(self, loss):
        grads = jt.grad(loss, self.params)
        scale = 1.0 / self.accumulation_steps
        for accumulator, grad in zip(self.grad_accum, grads):
            accumulator.update(accumulator + grad.float32() * scale)
        jt.sync(self.grad_accum)
        self.micro_step += 1
        del grads
        jt.clean_graph()
        jt.gc()

    def ready(self):
        return self.micro_step % self.accumulation_steps == 0

    def step(self):
        if not self.ready():
            raise RuntimeError("optimizer step requested before accumulation completes")

        flat = jt.concat([value.flatten() for value in self.grad_accum])
        if jt.in_mpi:
            flat = flat.mpi_all_reduce("mean")
        flat.sync()

        self.optimizer_step += 1
        beta1_power = self.beta1 ** self.optimizer_step
        beta2_power = self.beta2 ** self.optimizer_step
        step_size = self.lr / (1.0 - beta1_power)
        denominator_scale = math.sqrt(1.0 - beta2_power)
        offset = 0
        sync_values = []

        for index, parameter in enumerate(self.params):
            count = int(parameter.numel())
            grad = flat[offset : offset + count].reshape(parameter.shape)
            offset += count

            master = self.master[index]
            exp_avg = self.exp_avg[index]
            exp_avg_sq = self.exp_avg_sq[index]
            ema1 = self.ema1[index]
            ema2 = self.ema2[index]

            new_exp_avg = self.beta1 * exp_avg + (1.0 - self.beta1) * grad
            new_exp_avg_sq = (
                self.beta2 * exp_avg_sq + (1.0 - self.beta2) * grad * grad
            )
            denominator = jt.sqrt(new_exp_avg_sq) / denominator_scale + self.eps
            new_master = master * (1.0 - self.lr * self.weight_decay)
            new_master = new_master - step_size * new_exp_avg / denominator
            new_ema1 = (
                self.ema_decays[0] * ema1
                + (1.0 - self.ema_decays[0]) * new_master
            )
            new_ema2 = (
                self.ema_decays[1] * ema2
                + (1.0 - self.ema_decays[1]) * new_master
            )

            exp_avg.update(new_exp_avg)
            exp_avg_sq.update(new_exp_avg_sq)
            master.update(new_master)
            ema1.update(new_ema1)
            ema2.update(new_ema2)
            parameter.update(new_master.bfloat16())
            self.grad_accum[index].update(self.grad_accum[index] * 0.0)
            sync_values.extend(
                (
                    exp_avg,
                    exp_avg_sq,
                    master,
                    ema1,
                    ema2,
                    parameter,
                    self.grad_accum[index],
                )
            )

        if offset != int(flat.numel()):
            raise RuntimeError(f"gradient size mismatch: used {offset}, got {flat.numel()}")
        jt.sync(sync_values)
        del flat
        jt.clean_graph()
        jt.gc()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default="/mnt/nfs/home/xutianyi/datasets/imagenet-1k/train",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Optional Torch checkpoint to initialize from. If omitted, use "
            "the upstream JiT random-initialization rules."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/nfs/home/xutianyi/JiT/outputs/jittor_b16_two_node_benchmark",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--micro-batch", type=int, default=128)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--steps-per-round", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-epochs", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--monitor-script",
        default=(
            "/mnt/nfs/home/xutianyi/JiT/benchmarks/"
            "jit_b16_a100/monitor_resources.py"
        ),
    )
    return parser.parse_args()


def barrier():
    if jt.in_mpi:
        jt.compile_extern.mpi.mpi_barrier()


def write_event(handle, event):
    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    handle.flush()
    print(event["event"].upper(), json.dumps(event, ensure_ascii=False), flush=True)


def main():
    args = parse_args()
    if args.steps_per_round % args.accumulation_steps:
        raise ValueError("--steps-per-round must be divisible by accumulation steps")

    jt.flags.use_cuda = 1
    rank = jt.compile_extern.mpi.world_rank() if jt.in_mpi else 0
    world = jt.compile_extern.mpi.world_size() if jt.in_mpi else 1
    hostname = socket.gethostname()
    # All ranks must construct identical random model parameters. Dataset and
    # per-rank stochastic training streams are separated after model creation.
    jt.set_global_seed(args.seed)
    np.random.seed(args.seed + rank)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / f"events_{hostname}.jsonl"
    resources_path = output_dir / f"resources_{hostname}.csv"
    event_handle = events_path.open("w", buffering=1, encoding="utf-8")

    monitor = subprocess.Popen(
        [
            sys.executable,
            args.monitor_script,
            "--pid",
            str(os.getpid()),
            "--output",
            str(resources_path),
            "--interval",
            "1.0",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    total_started = time.perf_counter()
    try:
        dataset_started = time.perf_counter()
        dataset = ImageFolder(
            args.data_path,
            transform=ImageTransform(args.image_size),
        ).set_attrs(
            batch_size=args.micro_batch * world,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
        )
        dataset_seconds = time.perf_counter() - dataset_started
        dataset_size = len(dataset.imgs)
        full_micro_steps = dataset_size // (args.micro_batch * world)
        reference_samples = full_micro_steps * args.micro_batch * world

        model_started = time.perf_counter()
        model = TrainingDenoiser(image_size=args.image_size)
        if args.checkpoint:
            checkpoint = jt.load(args.checkpoint)
            state = checkpoint["model"]
            model.load_parameters(state)
            del state, checkpoint
            initialization = "checkpoint"
        else:
            initialization = "random_upstream_rules"
        gc.collect()
        jt.sync_all()
        jt.set_global_seed(args.seed + rank)

        for name, parameter in model.named_parameters():
            if name == "net.pos_embed":
                parameter.stop_grad()

        # Torch keeps FP32 model parameters while autocast executes selected
        # kernels in BF16. Jittor 1.3.11 has no equivalent BF16-autocast path,
        # so preserve the exact FP32 initialization as optimizer master/EMA
        # state before creating the BF16 forward copy.
        master_initial_values = {
            name: parameter.float32().copy().stop_grad()
            for name, parameter in model.named_parameters()
            if not parameter.is_stop_grad()
        }
        jt.sync(list(master_initial_values.values()))
        model.bfloat16()
        for name, parameter in model.named_parameters():
            if name == "net.pos_embed":
                parameter.stop_grad()
        model.train()
        jt.sync_all()
        model_seconds = time.perf_counter() - model_started

        optimizer_started = time.perf_counter()
        optimizer = MasterAdamW(
            list(model.named_parameters()),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=0.0,
            accumulation_steps=args.accumulation_steps,
            ema_decays=(0.9999, 0.9996),
            master_initial_values=master_initial_values,
        )
        del master_initial_values
        optimizer_seconds = time.perf_counter() - optimizer_started
        trainable_parameters = sum(
            int(parameter.numel())
            for _, parameter in model.named_parameters()
            if not parameter.is_stop_grad()
        )

        config = {
            "framework": f"jittor {jt.__version__}",
            "model": "JiT-B/16",
            "initialization": initialization,
            "trainable_parameters": trainable_parameters,
            "image_size": args.image_size,
            "precision": {
                "forward_parameters_and_activations": "bfloat16",
                "attention_qk_and_loss": "float32",
                "gradients_allreduce": "float32",
                "master_weights_optimizer_and_ema": "float32",
            },
            "optimizer": "AdamW",
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "learning_rate_target": args.lr,
            "warmup_epochs": args.warmup_epochs,
            "ema_decays": [0.9999, 0.9996],
            "micro_batch_per_rank": args.micro_batch,
            "global_micro_batch": args.micro_batch * world,
            "accumulation_steps": args.accumulation_steps,
            "effective_batch": args.micro_batch * world * args.accumulation_steps,
            "world_size": world,
            "dataset_size": dataset_size,
            "reference_samples_per_epoch": reference_samples,
            "full_micro_steps_per_epoch": full_micro_steps,
            "rounds": args.rounds,
            "steps_per_round": args.steps_per_round,
            "rank": rank,
            "hostname": hostname,
            "dataset_scan_seconds": dataset_seconds,
            "model_load_seconds": model_seconds,
            "optimizer_init_seconds": optimizer_seconds,
            "checkpoint": args.checkpoint,
            "data_path": args.data_path,
            "note": (
                "Benchmark only: real ImageNet training, FP32 master AdamW, "
                "dual EMA, no checkpoint write or online FID."
            ),
        }
        (output_dir / f"config_{hostname}.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        write_event(
            event_handle,
            {
                "event": "ready",
                "rank": rank,
                "world_size": world,
                "host": hostname,
                "dataset_seconds": dataset_seconds,
                "model_seconds": model_seconds,
                "optimizer_seconds": optimizer_seconds,
                "trainable_parameters": trainable_parameters,
            },
        )

        iterator = iter(dataset)
        total_micro_steps = 0
        total_samples = 0
        run_started = time.perf_counter()
        for round_index in range(args.rounds):
            barrier()
            round_started = time.perf_counter()
            round_loss = 0.0

            for step_in_round in range(args.steps_per_round):
                try:
                    images, labels = next(iterator)
                except StopIteration:
                    iterator = iter(dataset)
                    images, labels = next(iterator)

                images = images.float32() / 127.5 - 1.0
                labels = labels.int32()
                fractional_epoch = total_micro_steps / full_micro_steps
                optimizer.lr = args.lr * min(
                    fractional_epoch / args.warmup_epochs,
                    1.0,
                )

                loss = model(images, labels)
                loss_value = float(loss.item())
                if not math.isfinite(loss_value):
                    raise RuntimeError(f"non-finite loss: {loss_value}")
                optimizer.backward(loss)
                if optimizer.ready():
                    optimizer.step()

                total_micro_steps += 1
                total_samples += args.micro_batch * world
                round_loss += loss_value

                if (
                    step_in_round == 0
                    or (step_in_round + 1) % args.log_every == 0
                    or step_in_round + 1 == args.steps_per_round
                ):
                    elapsed = time.perf_counter() - round_started
                    write_event(
                        event_handle,
                        {
                            "event": "progress",
                            "rank": rank,
                            "host": hostname,
                            "round": round_index,
                            "micro_steps": step_in_round + 1,
                            "optimizer_steps": optimizer.optimizer_step,
                            "samples": (step_in_round + 1)
                            * args.micro_batch
                            * world,
                            "seconds": elapsed,
                            "samples_per_second": (
                                (step_in_round + 1)
                                * args.micro_batch
                                * world
                                / elapsed
                            ),
                            "loss": loss_value,
                            "lr": optimizer.lr,
                        },
                    )

            barrier()
            round_seconds = time.perf_counter() - round_started
            round_samples = args.steps_per_round * args.micro_batch * world
            write_event(
                event_handle,
                {
                    "event": "round_end",
                    "rank": rank,
                    "host": hostname,
                    "round": round_index,
                    "micro_steps": args.steps_per_round,
                    "optimizer_steps": (
                        args.steps_per_round // args.accumulation_steps
                    ),
                    "samples": round_samples,
                    "seconds": round_seconds,
                    "samples_per_second": round_samples / round_seconds,
                    "mean_loss": round_loss / args.steps_per_round,
                },
            )

        barrier()
        run_seconds = time.perf_counter() - run_started
        write_event(
            event_handle,
            {
                "event": "run_end",
                "rank": rank,
                "host": hostname,
                "rounds": args.rounds,
                "micro_steps": total_micro_steps,
                "optimizer_steps": optimizer.optimizer_step,
                "samples": total_samples,
                "seconds": run_seconds,
                "samples_per_second": total_samples / run_seconds,
                "total_seconds_including_setup": time.perf_counter()
                - total_started,
            },
        )
    finally:
        event_handle.close()
        monitor.terminate()
        try:
            monitor.wait(timeout=5)
        except subprocess.TimeoutExpired:
            monitor.kill()
            monitor.wait()


if __name__ == "__main__":
    main()
