"""Jittor ODE sampler matching the JiT inference defaults."""

import jittor as jt
from jittor import nn

from model_jit import JiT_B_16


class Denoiser(nn.Module):
    def __init__(
        self,
        img_size=256,
        num_classes=1000,
        steps=50,
        method="heun",
        cfg_scale=3.0,
        interval_min=0.1,
        interval_max=1.0,
        t_eps=0.05,
        noise_scale=1.0,
        cfg_batch=True,
    ):
        super().__init__()
        self.net = JiT_B_16(
            input_size=img_size,
            in_channels=3,
            num_classes=num_classes,
        )
        self.img_size = img_size
        self.num_classes = num_classes
        self.steps = steps
        self.method = method
        self.cfg_scale = cfg_scale
        self.cfg_interval = (interval_min, interval_max)
        self.t_eps = t_eps
        self.noise_scale = noise_scale
        self.cfg_batch = cfg_batch

    def _velocity(self, z, t, labels):
        denominator = jt.maximum(
            1.0 - t, jt.ones_like(t) * self.t_eps
        )
        null_labels = jt.ones_like(labels) * self.num_classes

        if self.cfg_batch:
            z_both = jt.concat((z, z), dim=0)
            t_both = jt.concat((t.flatten(), t.flatten()), dim=0)
            labels_both = jt.concat((labels, null_labels), dim=0)
            prediction = self.net(z_both, t_both, labels_both)
            x_cond, x_uncond = prediction.chunk(2, dim=0)
        else:
            flat_t = t.flatten()
            x_cond = self.net(z, flat_t, labels)
            x_cond.sync()
            v_cond = (x_cond - z) / denominator
            v_cond.sync()
            x_uncond = self.net(z, flat_t, null_labels)
            x_uncond.sync()
            v_uncond = (x_uncond - z) / denominator
            v_uncond.sync()

        if self.cfg_batch:
            v_cond = (x_cond - z) / denominator
            v_uncond = (x_uncond - z) / denominator
        low, high = self.cfg_interval
        interval_mask = (t < high) & (t > low if low else (t >= 0.0))
        scale = jt.where(
            interval_mask,
            jt.ones_like(t) * self.cfg_scale,
            jt.ones_like(t),
        )
        return v_uncond + scale * (v_cond - v_uncond)

    def _euler_step(self, z, t, t_next, labels):
        return z + (t_next - t) * self._velocity(z, t, labels)

    def _heun_step(self, z, t, t_next, labels):
        velocity_t = self._velocity(z, t, labels)
        euler = z + (t_next - t) * velocity_t
        velocity_next = self._velocity(euler, t_next, labels)
        return z + (t_next - t) * 0.5 * (velocity_t + velocity_next)

    def generate(self, labels, progress_callback=None, initial_noise=None):
        batch_size = labels.shape[0]
        if initial_noise is None:
            initial_noise = jt.randn(
                (batch_size, 3, self.img_size, self.img_size)
            )
        z = self.noise_scale * initial_noise

        for index in range(self.steps):
            t_value = index / float(self.steps)
            next_value = (index + 1) / float(self.steps)
            t = jt.ones((batch_size, 1, 1, 1)) * t_value
            t_next = jt.ones((batch_size, 1, 1, 1)) * next_value

            # The reference always uses Euler for its final step.
            if self.method == "heun" and index < self.steps - 1:
                z = self._heun_step(z, t, t_next, labels)
            elif self.method == "euler" or index == self.steps - 1:
                z = self._euler_step(z, t, t_next, labels)
            else:
                raise ValueError(f"Unknown sampling method: {self.method}")

            # Jittor is lazy. Synchronizing once per ODE step prevents a
            # 50-step graph from accumulating in memory.
            z.sync()
            if progress_callback is not None:
                progress_callback(index + 1, self.steps)
        return z
