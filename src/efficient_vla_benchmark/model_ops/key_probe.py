"""Read native post-RoPE keys without computing probabilities or replacing SDPA."""

from transformers.models.llama.modeling_llama import rotate_half


class NativeKeyProbe:
    def __init__(self, attention):
        self.handles = []
        self.clear()
        try:
            self.handles = [
                attention.register_forward_pre_hook(self._start, with_kwargs=True),
                attention.k_proj.register_forward_hook(self._capture),
                attention.register_forward_hook(self._finish, with_kwargs=True, always_call=True),
            ]
        except Exception:
            self.close()
            raise

    def clear(self):
        self.projected = self.keys = None

    def _start(self, module, args, kwargs):
        self.clear()
        cache = kwargs.get("past_key_value")
        if cache is not None and cache.get_seq_length(module.layer_idx):
            raise ValueError("Key probes support fresh prefill only")

    def _capture(self, module, args, output):
        self.projected = output

    def _finish(self, module, args, kwargs, output):
        try:
            if output is None:
                return
            value = self.projected
            if value is None or value.shape[0] != 1:
                raise ValueError("Expected batch-one native key projection")
            length = value.shape[1]
            keys = value.view(1, length, module.num_key_value_heads, module.head_dim).transpose(
                1, 2
            )
            cos, sin = module.rotary_emb(keys, kwargs["position_ids"])
            keys = keys * cos.unsqueeze(1) + rotate_half(keys) * sin.unsqueeze(1)
            self.keys = keys.transpose(1, 2).reshape(length, -1)
        finally:
            self.projected = None

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.clear()
