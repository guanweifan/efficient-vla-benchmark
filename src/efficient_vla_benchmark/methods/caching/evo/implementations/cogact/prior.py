"""Observe uncached closed-loop queries; feature dissimilarity is only a prior."""

from dataclasses import asdict

import numpy as np
import torch

from efficient_vla_benchmark.model_ops.forward_patch import ForwardPatches


class NativePrior:
    def __init__(self, host, queries_per_episode=2):
        if host.method is not None or queries_per_episode < 1:
            raise ValueError("EVO prior requires uncached native queries")
        self.host, self.limit = host, queries_per_episode
        self.net = host.model_access.action_denoiser
        self.branches = [m for block in self.net.blocks for m in (block.attn, block.mlp)]
        self.steps = host.parameters.num_ddim_steps
        self.total = np.zeros((len(self.branches), self.steps), dtype=np.float64)
        self.queries, self.identities = 0, []
        self.handles = []
        self.patches = ForwardPatches()
        self.before_episode(None)
        # CFG invokes net.forward directly and bypasses the module call hooks.
        # Wrap this instance exactly as the deployment cache does.
        self.patches.wrap(self.net, self._forward)
        self.handles.extend(
            m.register_forward_hook(self._capture(i)) for i, m in enumerate(self.branches)
        )

    def _forward(self, original):
        def forward(*args, **kwargs):
            self._before_step(self.net, args)
            result = original(*args, **kwargs)
            self._after_step(self.net, args, result)
            return result

        return forward

    def before_episode(self, context):
        self.episode_queries, self.step = 0, -1
        self.features = [[] for _ in self.branches]
        self.context = asdict(context) if context else None

    def _before_step(self, module, args):
        self.step = (self.step + 1) % self.steps

    def _capture(self, index):
        def hook(module, args, output):
            if self.episode_queries < self.limit:
                self.features[index].append(output.detach().float().flatten().clone())

        return hook

    def _after_step(self, module, args, output):
        if self.step != self.steps - 1:
            return
        if self.episode_queries < self.limit:
            if any(len(values) != self.steps for values in self.features):
                raise RuntimeError("Incomplete native activation trajectory")
            rows = []
            for values in self.features:
                matrix = torch.stack(values)
                if not torch.isfinite(matrix).all():
                    raise ValueError("Nonfinite native residual features")
                matrix = torch.nn.functional.normalize(matrix, dim=1)
                similarity = (matrix @ matrix.T).clamp(-1, 1)
                dissimilarity = 1 - similarity
                dissimilarity.fill_diagonal_(0)
                rows.append((dissimilarity.sum(1) / (self.steps - 1)).cpu().numpy())
            self.total += np.stack(rows)
            self.queries += 1
            self.identities.append(dict(context=self.context, query=self.episode_queries))
            self.features = [[] for _ in self.branches]
        self.episode_queries += 1

    def result(self):
        if not self.queries:
            raise ValueError("No native rollout features were observed")
        return self.total / self.queries

    def close(self):
        self.patches.close()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.features.clear()
