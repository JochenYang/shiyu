"""
CTC forced alignment for accurate subtitle timestamps.

Uses Viterbi decoding to align external text onto CTC logits from
SenseVoice ONNX. Produces per-character frame positions that map
directly to real audio timestamps.

Algorithm:
  CTC-Segmentation style forced alignment. Given log-probabilities
  from a CTC model and a target text sequence, finds the optimal
  frame-level path through the CTC trellis using dynamic programming.
"""

import numpy as np
import sentencepiece as spm
from typing import List, Tuple, Optional

# SenseVoice-specific constants
_BLANK_ID = 0       # SenseVoice uses <pad> as blank token (id=0)
_SIL_ID = 1         # <sos> / silence token (skipped in alignment)
_EOS_ID = 2         # <eos> token (skipped)


def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable log-softmax."""
    x_max = x.max(axis=axis, keepdims=True)
    shifted = x - x_max
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def ctc_viterbi_align(logits: np.ndarray,
                      token_ids: List[int],
                      blank_id: int = _BLANK_ID) -> List[Tuple[int, int, int]]:
    """CTC Viterbi forced alignment in pure numpy.

    Aligns a sequence of token IDs onto CTC logits via dynamic programming.

    Args:
        logits: Raw model output, shape [T, V] (T frames, V vocabulary).
        token_ids: Target token ID sequence to align.
        blank_id: Index of the CTC blank token.

    Returns:
        List of (token_id, start_frame, end_frame) triples.
        start_frame (inclusive) and end_frame (exclusive) are CTC frame indices.
        Skip tokens with start_frame >= end_frame (they had no time coverage).
    """
    T = logits.shape[0]
    if not token_ids:
        return []

    # Convert logits to log-probabilities
    log_probs = _log_softmax(logits, axis=-1)  # [T, V]

    # Augment target: blank + (token, blank) × n
    # CTC formulation: insert blanks between every token
    L = len(token_ids)
    S = 2 * L + 1  # augmented sequence length

    # Build augmented target array
    aug_target = np.full(S, blank_id, dtype=np.int32)
    for i, tid in enumerate(token_ids):
        aug_target[2 * i + 1] = tid

    # Dynamic programming trellis: [T+1, S]
    # We use log-space throughout
    NEG_INF = -1e30
    trellis = np.full((T + 1, S), NEG_INF, dtype=np.float64)
    trellis[0, 0] = 0.0  # start at first blank

    # Backpointers for path reconstruction
    backpointers = np.zeros((T + 1, S), dtype=np.int32)

    for t in range(1, T + 1):
        # Log prob of current frame for the augmented target at each s
        lp = log_probs[t - 1, aug_target]  # [S]

        for s in range(min(S, 2 * t - 1)):
            # Three possible transitions:
            # 1. Stay at same state (from t-1, s)
            candidates = [trellis[t - 1, s]]

            # 2. Advance from previous state (from t-1, s-1)
            if s > 0:
                candidates.append(trellis[t - 1, s - 1])

            # 3. Skip a blank (from t-1, s-2)
            # Only allowed if current and s-2 tokens are different
            if s >= 2 and aug_target[s] != aug_target[s - 2]:
                candidates.append(trellis[t - 1, s - 2])

            # Pick the best
            best_idx = int(np.argmax(candidates))
            trellis[t, s] = candidates[best_idx] + lp[s]
            backpointers[t, s] = best_idx

    # Backtrack from the last blank state (S-1)
    # Or from the last non-blank if it's better
    final_state = S - 1  # end at final blank
    if trellis[T, S - 2] > trellis[T, final_state]:
        final_state = S - 2

    path = np.zeros(T, dtype=np.int32)
    s = final_state
    for t in range(T, 0, -1):
        path[t - 1] = s
        # Follow backpointer
        bp = backpointers[t, s]
        if bp == 0:
            s = s  # stay
        elif bp == 1:
            s = s - 1  # advance
        else:
            s = s - 2  # skip blank

    # Convert path to per-token frame boundaries
    # For each token in the original sequence, find its start and end frames
    result = []
    for i, tid in enumerate(token_ids):
        state_pos = 2 * i + 1  # position of this token in augmented sequence
        frames = np.where(path == state_pos)[0]
        if len(frames) > 0:
            start_f = int(frames[0])
            end_f = int(frames[-1]) + 1
            result.append((tid, start_f, end_f))
        else:
            # Token not found - fallback: estimate position proportionally
            prop_start = int(T * (i / max(L, 1)))
            prop_end = int(T * ((i + 1) / max(L, 1)))
            result.append((tid, max(0, prop_start), min(T, prop_end)))

    return result


def align_text(logits: np.ndarray,
               text: str,
               sp_model: spm.SentencePieceProcessor,
               blank_id: int = _BLANK_ID) -> List[Tuple[str, int, int]]:
    """Full text-to-frame alignment pipeline.

    Tokenizes text, runs Viterbi alignment, returns character-level
    frame positions.

    Args:
        logits: Raw model output, shape [T, V].
        text: Target text to align (may include punctuation).
        sp_model: Loaded SentencePiece processor matching the model.
        blank_id: CTC blank token index.

    Returns:
        List of (character, start_frame, end_frame) triples.
        Characters are single Unicode characters (CJK) or subword pieces.
    """
    # Tokenize text
    token_ids = sp_model.EncodeAsIds(text)

    # Run Viterbi alignment
    aligned = ctc_viterbi_align(logits, token_ids, blank_id)

    # Convert token-level to character-level
    # For CJK BPE models, most tokens are single characters.
    # Expand subword tokens to per-character granularity.
    char_level = []
    for tid, start_f, end_f in aligned:
        token_str = sp_model.IdToPiece(tid)
        # try decode single char if possible
        try:
            decoded = sp_model.DecodeIds([tid])
        except Exception:
            decoded = token_str

        # For CJK characters, decoded is usually a single character
        # For BPE subwords, it might be multiple characters
        if len(decoded) == 1:
            char_level.append((decoded, start_f, end_f))
        else:
            # Distribute frames evenly across characters
            chars = list(decoded)
            if len(chars) > 1 and end_f > start_f:
                seg_size = max(1, (end_f - start_f) // len(chars))
                for ci, c in enumerate(chars):
                    cs = start_f + ci * seg_size
                    ce = min(end_f, cs + seg_size)
                    char_level.append((c, cs, ce))
            else:
                char_level.append((decoded, start_f, end_f))

    return char_level


def align_text_to_timestamps(logits: np.ndarray,
                              text: str,
                              sp_model: spm.SentencePieceProcessor,
                              frame_to_sec: float = 0.06,
                              ctc_delay_sec: float = 0.15) -> List[dict]:
    """Align text to audio timestamps using CTC forced alignment.

    High-level API: given CTC logits and a reference text, produce
    per-character timestamps suitable for subtitle segmentation.

    Args:
        logits: CTC model output, shape [T, V].
        text: Reference text (from any ASR engine).
        sp_model: SentencePiece processor.
        frame_to_sec: Seconds per CTC frame (SenseVoice: 60ms).
        ctc_delay_sec: Compensation for CTC peak delay.

    Returns:
        List of dicts: [{"char": str, "start": float, "end": float}, ...]
    """
    char_level = align_text(logits, text, sp_model, blank_id=_BLANK_ID)

    timestamps = []
    for c, start_f, end_f in char_level:
        if c.strip():
            timestamps.append({
                "char": c,
                "start": max(0, start_f * frame_to_sec - ctc_delay_sec),
                "end": end_f * frame_to_sec,
            })
    return timestamps


def load_sp_model(model_dir: str) -> Optional[spm.SentencePieceProcessor]:
    """Load SenseVoice's SentencePiece BPE model.

    Args:
        model_dir: Path to model directory containing .bpe.model file.

    Returns:
        Loaded SentencePieceProcessor, or None if not found.
    """
    import os
    for fname in os.listdir(model_dir):
        if fname.endswith(".bpe.model"):
            path = os.path.join(model_dir, fname)
            sp = spm.SentencePieceProcessor()
            sp.Load(path)
            return sp
    return None
