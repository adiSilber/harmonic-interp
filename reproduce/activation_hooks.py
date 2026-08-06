"""
Activation capture hooks for Aria model inference.

Vendored from the aria library (aria/inference/activation_hooks.py) so the
reproduce/ pipeline is self-contained and frozen against upstream changes.
The hook mechanics are byte-identical to the library version; the only change
here is that `save()` accepts `hidden_states_positions` to write the residual
stream for a subset of positions (the per-layer patching experiment only needs
it at the patch position).

Captures:
- KV cache (keys, values) from each transformer layer
- Hidden state activations at each layer
- Logits from the LM head

Usage:
    from activation_hooks import ActivationCollector

    collector = ActivationCollector(model)
    collector.register_hooks()

    # Run inference...

    # Get captured data
    data = collector.get_captured_data()
    collector.save(output_path)
    collector.clear()
"""

import torch
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CapturedActivations:
    """Container for captured activations at a specific token position."""
    token_idx: int
    token_id: int

    # Hidden states per layer: layer_idx -> tensor of shape (d_model,)
    hidden_states: dict = field(default_factory=dict)

    # KV cache per layer: layer_idx -> {'k': tensor, 'v': tensor}
    # Shape per tensor: (n_heads, 1, head_dim) — only the KV for this specific position
    kv_cache: dict = field(default_factory=dict)

    # Logits at this position: shape (vocab_size,)
    logits: Optional[torch.Tensor] = None

    # Probabilities (after softmax with temperature).
    # Convention: logits/probs stored at capture position p are the model's
    # next-token distribution computed AT p — i.e. they predict the token at
    # p+1 — while token_id is the token at p itself. This holds everywhere
    # these are saved (probs.pt / logits.pt), including the patch position.
    probs: Optional[torch.Tensor] = None


class ActivationCollector:
    """
    Collects activations from an Aria TransformerLM model during inference.

    Can capture:
    - Hidden states after each transformer layer
    - KV cache values from each layer
    - Final logits and probabilities
    """

    def __init__(
        self,
        model,
        capture_hidden_states: bool = True,
        capture_kv_cache: bool = True,
        capture_logits: bool = True,
        capture_positions: Optional[list[int]] = None,
        device: str = "cpu",
        full_kv_positions: Optional[list[int]] = None,
    ):
        """
        Args:
            model: TransformerLM model instance
            capture_hidden_states: Whether to capture hidden states per layer
            capture_kv_cache: Whether to capture KV cache values
            capture_logits: Whether to capture logits
            capture_positions: If specified, only capture at these token positions.
                              If None, capture all positions.
            device: Device to store captured tensors ('cpu' to save GPU memory)
            full_kv_positions: Positions where the full KV history (0..token_idx+1)
                              should be saved instead of just the single-position slice.
        """
        self.model = model
        self.capture_hidden_states = capture_hidden_states
        self.capture_kv_cache = capture_kv_cache
        self.capture_logits = capture_logits
        self.capture_positions = set(capture_positions) if capture_positions else None
        self.device = device
        self.full_kv_positions = set(full_kv_positions) if full_kv_positions else set()

        self.hooks = []
        self.captured_data: list[CapturedActivations] = []

        # Temporary storage during forward pass
        self._current_hidden_states = {}
        self._current_input_pos = None
        self._current_token_ids = None

    def register_hooks(self):
        """Register forward hooks on model layers."""
        self.clear_hooks()

        # Hook on each transformer block to capture hidden states
        if self.capture_hidden_states:
            for layer_idx, layer in enumerate(self.model.model.encode_layers):
                hook = layer.register_forward_hook(
                    self._make_hidden_state_hook(layer_idx)
                )
                self.hooks.append(hook)

        # Hook on the LM head to capture logits
        if self.capture_logits:
            hook = self.model.lm_head.register_forward_hook(self._logits_hook)
            self.hooks.append(hook)

    def _make_hidden_state_hook(self, layer_idx: int):
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            # output shape: (batch_size, seq_len, d_model)
            # Store the output for later processing
            self._current_hidden_states[layer_idx] = output.detach().clone()
        return hook

    def _logits_hook(self, module, input, output):
        """Hook to capture logits from LM head."""
        # output shape: (batch_size, seq_len, vocab_size)
        self._current_logits = output.detach()

    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear(self):
        """Clear all captured data."""
        self.captured_data = []
        self._current_hidden_states = {}
        self._current_logits = None

    def set_capture_positions(self, positions: list[int]):
        """Update which token positions to capture."""
        self.capture_positions = set(positions) if positions else None

    def should_capture(self, token_idx: int) -> bool:
        """Check if we should capture data at this position."""
        if self.capture_positions is None:
            return True
        return token_idx in self.capture_positions

    def capture_step(
        self,
        token_idx: int,
        token_id: int,
        probs: Optional[torch.Tensor] = None,
        batch_idx: int = 0,
        seq_position: int = -1,
        clear_after: bool = True,
    ):
        """
        Call this after each forward pass to capture and store activations.

        Args:
            token_idx: Current token position in the sequence (for labeling)
            token_id: The token ID at this position
            probs: Optional probability distribution (after temperature)
            batch_idx: Which batch element to capture (default 0)
            seq_position: Position in the current hidden states tensor to extract.
                         Use -1 for last position (default, for single-token decode).
                         Use actual index for prompt positions during prefill.
            clear_after: Whether to clear temp storage after capture (default True).
                        Set to False when capturing multiple positions from same forward pass.
        """
        if not self.should_capture(token_idx):
            return

        captured = CapturedActivations(
            token_idx=token_idx,
            token_id=token_id,
        )

        # Capture hidden states
        if self.capture_hidden_states:
            if self._current_hidden_states:
                for layer_idx, hidden in self._current_hidden_states.items():
                    # hidden shape: (batch_size, seq_len, d_model)
                    # Extract the specified position
                    captured.hidden_states[layer_idx] = hidden[batch_idx, seq_position].to(self.device).clone()
            else:
                print(f"  WARNING: No hidden states captured for position {token_idx}")

        # Capture KV cache
        if self.capture_kv_cache:
            full_kv = token_idx in self.full_kv_positions
            for layer_idx, layer in enumerate(self.model.model.encode_layers):
                if layer.kv_cache is not None:
                    if full_kv:
                        # Full history: shape (n_heads, token_idx+1, head_dim)
                        k = layer.kv_cache.k_cache[batch_idx, :, :token_idx+1].to(self.device).clone()
                        v = layer.kv_cache.v_cache[batch_idx, :, :token_idx+1].to(self.device).clone()
                    else:
                        # Single position: shape (n_heads, 1, head_dim)
                        k = layer.kv_cache.k_cache[batch_idx, :, token_idx:token_idx+1].to(self.device).clone()
                        v = layer.kv_cache.v_cache[batch_idx, :, token_idx:token_idx+1].to(self.device).clone()
                    captured.kv_cache[layer_idx] = {'k': k, 'v': v}

        # Capture logits
        if self.capture_logits:
            if hasattr(self, '_current_logits') and self._current_logits is not None:
                # logits shape: (batch_size, seq_len, vocab_size)
                captured.logits = self._current_logits[batch_idx, seq_position].to(self.device).clone()
            else:
                print(f"  WARNING: No logits captured for position {token_idx}")

        # Store probs if provided
        if probs is not None:
            captured.probs = probs[batch_idx].to(self.device).clone()

        self.captured_data.append(captured)

        # Clear temporary storage (only if requested)
        if clear_after:
            self._current_hidden_states = {}
            self._current_logits = None

    def get_captured_data(self) -> list[CapturedActivations]:
        """Return all captured activations."""
        return self.captured_data

    def save(self, output_path: str, save_tensors: bool = True, tokenizer=None,
             hidden_states_positions=None):
        """
        Save captured data to disk.

        Args:
            output_path: Base path for output files
            save_tensors: If True, save tensor data as .pt files
            tokenizer: If provided, add human-readable token_translation to metadata
            hidden_states_positions: If given (a set/list of token positions), only
                write hidden_states.pt for those positions. None = write for all.
                (Per-layer patching only needs the residual stream at the patch
                position; logits/probs are still saved for every captured position.)
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        hs_positions = set(hidden_states_positions) if hidden_states_positions is not None else None

        # Save metadata
        token_ids = [c.token_id for c in self.captured_data]
        metadata = {
            'num_captures': len(self.captured_data),
            'capture_positions': [c.token_idx for c in self.captured_data],
            'token_ids': token_ids,
        }
        if tokenizer is not None:
            from utils import build_token_translation
            metadata['token_translation'] = build_token_translation(token_ids, tokenizer)

        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        if save_tensors:
            for i, captured in enumerate(self.captured_data):
                capture_dir = output_path / f'position_{captured.token_idx}'
                capture_dir.mkdir(exist_ok=True)

                # Save hidden states (only at requested positions, if filtered)
                if captured.hidden_states and (hs_positions is None or captured.token_idx in hs_positions):
                    hidden_dict = {f'layer_{k}': v for k, v in captured.hidden_states.items()}
                    torch.save(hidden_dict, capture_dir / 'hidden_states.pt')

                # Save KV cache
                if captured.kv_cache:
                    kv_dict = {}
                    for layer_idx, kv in captured.kv_cache.items():
                        kv_dict[f'layer_{layer_idx}_k'] = kv['k']
                        kv_dict[f'layer_{layer_idx}_v'] = kv['v']
                    torch.save(kv_dict, capture_dir / 'kv_cache.pt')

                # Save logits
                if captured.logits is not None:
                    torch.save(captured.logits, capture_dir / 'logits.pt')

                # Save probs
                if captured.probs is not None:
                    torch.save(captured.probs, capture_dir / 'probs.pt')

