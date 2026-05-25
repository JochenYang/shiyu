"""Diagnose VAD chunking behavior for a specific audio file.
Shows how VAD splits the audio, the chunk boundaries, and highlights
potential issues for continuous audio (songs, rap, etc.).
"""
import sys
import os

# Ensure backend is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import librosa
import numpy as np


def find_energy_dip(waveform, target_idx, sample_rate, window_sec=1.5, hop_length=512):
    """Simulate SenseVoiceEngine._find_energy_dip"""
    window_samples = int(window_sec * sample_rate)
    search_start = max(0, target_idx - window_samples)
    search_end = min(len(waveform), target_idx + window_samples)
    best_idx = target_idx
    best_energy = float("inf")
    pos = search_start
    while pos + hop_length <= search_end:
        frame = waveform[pos:pos + hop_length]
        energy = np.sqrt(np.mean(frame ** 2))
        if energy < best_energy:
            best_energy = energy
            best_idx = pos
        pos += hop_length
    return best_idx


def diagnose(audio_path: str, top_db: float = 35, max_chunk_sec: float = 28.0, min_chunk_sec: float = 8.0):
    sample_rate = 16000

    print(f"Loading: {audio_path}")
    waveform, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    total_duration = len(waveform) / sr
    print(f"Duration: {total_duration:.2f}s ({len(waveform)} samples @ {sr}Hz)")

    # Step 1: VAD intervals
    intervals = librosa.effects.split(waveform, top_db=top_db, frame_length=2048, hop_length=512)
    print(f"\nVAD intervals (top_db={top_db}): {len(intervals)} found")
    if len(intervals) <= 3:
        for i, (s, e) in enumerate(intervals):
            print(f"  Interval {i}: {s/sr:.3f}s - {e/sr:.3f}s ({e-s} samples, {(e-s)/sr:.3f}s)")

    # Total detected non-silence duration
    total_nonsilent = sum(e - s for s, e in intervals) / sr
    print(f"  Total non-silent: {total_nonsilent:.2f}s ({total_nonsilent/total_duration*100:.1f}%)")

    # Step 2: Compute silences
    silences = []
    last_end = 0
    for start, end in intervals:
        if start > last_end:
            silences.append((last_end, start))
        last_end = end
    if last_end < len(waveform):
        silences.append((last_end, len(waveform)))

    print(f"\nSilence gaps: {len(silences)}")
    if len(silences) <= 10:
        for i, (s, e) in enumerate(silences):
            dur = (e - s) / sr
            print(f"  Silence {i}: {s/sr:.3f}s - {e/sr:.3f}s ({dur:.3f}s)")

    # Step 3: Simulate chunk creation
    chunks = []
    current_start = 0
    for sil_start, sil_end in silences:
        cut_point = (sil_start + sil_end) // 2
        if (cut_point - current_start) / sr > min_chunk_sec:
            chunks.append((current_start, cut_point))
            current_start = cut_point

    if current_start < len(waveform):
        if chunks and (len(waveform) - current_start) / sr < 2.0:
            prev_start, _ = chunks.pop()
            chunks.append((prev_start, len(waveform)))
        else:
            chunks.append((current_start, len(waveform)))

    if not chunks:
        chunks = [(0, len(waveform))]

    print(f"\nStep 3 chunks (before hard limit): {len(chunks)}")
    for i, (s, e) in enumerate(chunks):
        print(f"  Chunk {i}: {s/sr:.3f}s - {e/sr:.3f}s ({(e-s)/sr:.3f}s)")

    # Step 4: Hard limit — OLD: blind uniform split
    max_samples = int(max_chunk_sec * sr)
    old_final_chunks = []
    for c_start, c_end in chunks:
        if c_end - c_start <= max_samples:
            old_final_chunks.append((c_start, c_end))
        else:
            t = c_start
            while t < c_end:
                next_t = min(t + max_samples, c_end)
                old_final_chunks.append((t, next_t))
                t = next_t

    print(f"\nStep 4 (OLD) — blind uniform split <= {max_chunk_sec}s: {len(old_final_chunks)} chunks")
    for i, (s, e) in enumerate(old_final_chunks):
        print(f"  Chunk {i}: {s/sr:.3f}s - {e/sr:.3f}s ({(e-s)/sr:.3f}s)")

    # Step 4: Hard limit — NEW: energy-dip aware split (matches actual code)
    min_split_samples = int(5.0 * sr)
    print(f"\nStep 4 (NEW) — energy-dip aware split <= {max_chunk_sec}s:")
    new_final_chunks = []
    for c_start, c_end in chunks:
        if c_end - c_start <= max_samples:
            new_final_chunks.append((c_start, c_end))
        else:
            t = c_start
            while t < c_end:
                remaining = c_end - t
                if remaining <= max_samples:
                    new_final_chunks.append((t, c_end))
                    break
                target = t + max_samples
                split_pos = find_energy_dip(waveform, target, sr, window_sec=1.5, hop_length=512)
                # Clamp: don't exceed boundary, keep at least 5s for trailing chunk
                split_pos = min(split_pos, c_end - min_split_samples)
                if split_pos <= t:
                    split_pos = min(t + max_samples, c_end)
                new_final_chunks.append((t, split_pos))
                t = split_pos

    for i, (s, e) in enumerate(new_final_chunks):
        print(f"  Chunk {i}: {s/sr:.3f}s - {e/sr:.3f}s ({(e-s)/sr:.3f}s)")

    # Show boundary shifts
    print(f"\n  Boundary comparison (OLD -> NEW, shift in seconds):")
    for i in range(min(len(old_final_chunks), len(new_final_chunks))):
        old_e = old_final_chunks[i][1] / sr
        new_e = new_final_chunks[i][1] / sr
        shift = new_e - old_e
        marker = " !!!" if abs(shift) > 0.5 else ""
        print(f"    Chunk {i} end: {old_e:.3f}s -> {new_e:.3f}s (shift: {shift:+.3f}s){marker}")

    # Warning flags
    print(f"\n=== Analysis ===")
    if len(intervals) <= 1:
        print("WARNING: VAD found <2 intervals (continuous audio, no silence gaps detected)")
        print("  Hard limit will split blind/energy-dip at ~", max_chunk_sec, "s intervals")
        print("  Smart split tries to find quietest point near each boundary")

        # Count large shifts (improvements)
        large_shifts = sum(1 for i in range(min(len(old_final_chunks), len(new_final_chunks)))
                          if abs(old_final_chunks[i][1] - new_final_chunks[i][1]) / sr > 0.5)
        if large_shifts > 0:
            print(f"  Smart split shifted {large_shifts}/{len(new_final_chunks)} boundaries by >0.5s")
    elif len(intervals) > 100:
        print("WARNING: VAD found very many intervals (>100), audio has high-frequency energy variation")
    else:
        print("VAD found natural split points - good for chunking")

    if len(silences) == 0:
        print("WARNING: Zero silence gaps found")
        print("  All audio is considered non-silent (music/noise across entire duration)")

    # Check chunk uniformity
    for name, fcs in [("OLD", old_final_chunks), ("NEW", new_final_chunks)]:
        chunk_durs = [(e - s) / sr for s, e in fcs]
        if chunk_durs:
            max_dur = max(chunk_durs)
            min_dur = min(chunk_durs)
            if max_dur - min_dur < 0.5 and len(fcs) > 1 and len(intervals) <= 1:
                print(f"{name}: Chunks nearly uniform ({min_dur:.1f}s - {max_dur:.1f}s)")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Administrator\Desktop\宝石Gem - 枪火.mp3"
    diagnose(path)
