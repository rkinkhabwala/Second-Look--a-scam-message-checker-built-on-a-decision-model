"""Hugging Face causal-LM scorer for engine A (the only part that touches torch).

Every question prompt starts with the same system prompt and message, so we:
1. tokenize each full chat prompt,
2. run the longest common token prefix through the model once (KV cache),
3. repeat that cache across the batch and run every question's suffix at once,
4. read logits only at each row's final position, for the option-letter tokens.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from second_look.engines.device import pick_device, pick_dtype
from second_look.engines.llm_scoring import LETTERS


def common_prefix_len(seqs: list[list[int]]) -> int:
    """Length of the shared token prefix, leaving at least one token in every sequence."""
    limit = min(len(s) for s in seqs) - 1
    n = 0
    while n < limit and all(s[n] == seqs[0][n] for s in seqs):
        n += 1
    return n


class HFOptionScorer:
    def __init__(self, repo_id: str, revision: str, device: str | None = None):
        self.model_id = f"{repo_id}@{revision[:8]}"
        self.repo_id = repo_id
        self.revision = revision
        self.device = pick_device(device)
        self.dtype = pick_dtype(self.device)
        self._model = None
        self._tok = None
        self._letter_ids: list[int] = []

    def load(self) -> None:
        if self._model is not None:
            return
        self._tok = AutoTokenizer.from_pretrained(self.repo_id, revision=self.revision)
        model = AutoModelForCausalLM.from_pretrained(self.repo_id, revision=self.revision, dtype=self.dtype)
        self._model = model.to(self.device).eval()
        ids = []
        for letter in LETTERS:
            toks = self._tok.encode(letter, add_special_tokens=False)
            if len(toks) != 1:
                raise RuntimeError(f"Option letter {letter!r} is not a single token for {self.repo_id}")
            ids.append(toks[0])
        self._letter_ids = ids

    def prompt_ids(self, system: str, context: str, suffix: str) -> list[int]:
        text = self._tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": f"{context}\n\n{suffix}"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # ignored by non-thinking models; required for hybrid Qwen3
        )
        return self._tok(text, add_special_tokens=False).input_ids

    @torch.inference_mode()
    def letter_logits(self, system: str, context: str, suffixes: list[str], n_options: list[int]) -> list[list[float]]:
        self.load()
        model, dev = self._model, self.device
        seqs = [self.prompt_ids(system, context, s) for s in suffixes]
        n = common_prefix_len(seqs)
        decoder, head = model.get_decoder(), model.get_output_embeddings()

        prefix = torch.tensor([seqs[0][:n]], device=dev)
        cache = decoder(input_ids=prefix, use_cache=True).past_key_values
        cache.batch_repeat_interleave(len(seqs))

        tails = [s[n:] for s in seqs]
        width = max(len(t) for t in tails)
        pad = self._tok.pad_token_id if self._tok.pad_token_id is not None else 0
        input_ids = torch.tensor([t + [pad] * (width - len(t)) for t in tails], device=dev)
        attention = torch.zeros((len(seqs), n + width), dtype=torch.long, device=dev)
        attention[:, :n] = 1
        for i, t in enumerate(tails):
            attention[i, n : n + len(t)] = 1  # right padding: real tokens never attend to pads
        positions = (torch.arange(width, device=dev) + n).unsqueeze(0).expand(len(seqs), -1)

        hidden = decoder(input_ids=input_ids, attention_mask=attention, position_ids=positions,
                         past_key_values=cache, use_cache=True).last_hidden_state
        last = torch.tensor([len(t) - 1 for t in tails], device=dev)
        logits = head(hidden[torch.arange(len(seqs), device=dev), last]).float()
        return [logits[i, self._letter_ids[:k]].tolist() for i, k in enumerate(n_options)]

    @torch.inference_mode()
    def letter_logits_naive(self, system: str, context: str, suffixes: list[str], n_options: list[int]) -> list[list[float]]:
        """Reference implementation (one full forward pass per question), used to verify the fast path."""
        self.load()
        out = []
        for suffix, k in zip(suffixes, n_options):
            ids = torch.tensor([self.prompt_ids(system, context, suffix)], device=self.device)
            logits = self._model(input_ids=ids).logits[0, -1].float()
            out.append(logits[self._letter_ids[:k]].tolist())
        return out
