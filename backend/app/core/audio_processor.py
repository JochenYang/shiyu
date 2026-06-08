"""
Audio Preprocessing for SenseVoice ONNX inference.
Steps:
  1. Load audio via librosa (supports video via ffmpeg extraction)
  2. Compute fbank using kaldi_native_fbank (matches SenseVoice training)
  3. Apply LFR (Low Frame Rate) stacking with left-context padding
  4. Apply CMVN (Cepstral Mean and Variance Normalization) from am.mvn
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import librosa
import kaldi_native_fbank as knf


def _run_hidden(cmd, **kwargs):
    """Run subprocess without flashing a console window on Windows."""
    opts = dict(kwargs)
    if sys.platform == "win32":
        opts.setdefault("creationflags", subprocess.CREATE_NO_WINDOW)
    subprocess.run(cmd, **opts)


def find_ffmpeg() -> str:
    """Locate ffmpeg executable.

    Search order:
      1. backend/bin/ffmpeg (or .exe on Windows) — bundled with app
      2. ~/.shiyu/bin/ffmpeg — auto-downloaded by ffmpeg_downloader.py
      3. FFMPEG_PATH environment variable
      4. System PATH
      5. Windows: known WinGet installation path (fallback)
      6. Trigger background download to ~/.shiyu/bin/
    Returns the path string, or 'ffmpeg' as last resort.
    """
    exe_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"

    # 1. Bundled binary (next to backend/ in backend/bin/)
    backend_dir = Path(__file__).resolve().parent.parent.parent
    bundled = backend_dir / "bin" / exe_name
    if bundled.is_file():
        return str(bundled)

    # 2. User data dir (~/.shiyu/bin/) — auto-downloaded
    shiyu_bin = Path.home() / ".shiyu" / "bin" / exe_name
    if shiyu_bin.is_file():
        return str(shiyu_bin)

    # 3. Environment variable override
    env_ffmpeg = os.environ.get("FFMPEG_PATH")
    if env_ffmpeg:
        try:
            subprocess.run(
                [env_ffmpeg, "-version"],
                capture_output=True, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            return env_ffmpeg
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # 4. System PATH
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        return "ffmpeg"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 5. Windows: WinGet fallback
    if sys.platform == "win32":
        winget_path = (
            r"C:\Users\Administrator\AppData\Local\Microsoft\WinGet\Packages"
            r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
            r"\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"
        )
        if os.path.exists(winget_path):
            return winget_path

    # 6. Trigger background download to ~/.shiyu/bin/
    try:
        from app.core.ffmpeg_downloader import ensure_ffmpeg
        ensure_ffmpeg()
        # Return the expected path even if download hasn't finished yet
        return str(shiyu_bin)
    except Exception:
        pass

    return "ffmpeg"  # Let it fail naturally


def extract_audio_from_video(video_path: str, output_wav: str = None, sample_rate: int = 16000) -> str:
    """Extract audio from video using ffmpeg. Returns path to wav file.

    Raises:
        RuntimeError: if ffmpeg is not found, the conversion fails, or the
            output wav file is empty. The error message includes the ffmpeg
            stderr (truncated) so the user can diagnose codec/container issues
            without grep-ing logs.
    """
    if output_wav is None:
        output_wav = tempfile.mktemp(suffix=".wav")

    ffmpeg_exe = find_ffmpeg()
    if ffmpeg_exe == "ffmpeg":
        # Heuristic: the literal string "ffmpeg" is returned when no bundled
        # binary, no env var, no PATH match, and no WinGet fallback found one.
        # We don't know whether the next subprocess.run() will actually find
        # one in PATH, so we let it run and surface a real error if it fails.
        # But we do emit a hint so the user knows fallback was used.
        print("[audio] Note: using system ffmpeg from PATH (no bundled binary found)")

    cmd = [
        ffmpeg_exe, "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ac", "1", "-ar", str(sample_rate),
        output_wav
    ]
    try:
        result = subprocess.run(
            cmd, check=False,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"ffmpeg executable not found at '{ffmpeg_exe}'. "
            f"Install ffmpeg or set FFMPEG_PATH env var. "
            f"Underlying error: {e}"
        )

    if result.returncode != 0:
        stderr = (result.stderr or b"").decode("utf-8", errors="replace")
        # ffmpeg prints status to stderr; trim to last 800 chars for readability
        stderr_tail = stderr.strip()[-800:] if stderr.strip() else "(no stderr)"
        raise RuntimeError(
            f"ffmpeg failed to extract audio from video. "
            f"ffmpeg={ffmpeg_exe}, video={video_path}, output={output_wav}, "
            f"returncode={result.returncode}. "
            f"ffmpeg stderr (last 800 chars): {stderr_tail}"
        )

    # Verify the output file is non-empty (ffmpeg can exit 0 but produce a
    # zero-byte file if the input has no audio track).
    if not os.path.exists(output_wav):
        raise RuntimeError(
            f"ffmpeg reported success but output wav does not exist: {output_wav}. "
            f"Check that the input video has an audio track."
        )
    if os.path.getsize(output_wav) < 1024:  # <1 KB is almost certainly empty WAV
        raise RuntimeError(
            f"ffmpeg produced a suspiciously small wav file ({os.path.getsize(output_wav)} bytes). "
            f"Input video may have no audio track or use an unsupported codec. "
            f"Output: {output_wav}"
        )

    return output_wav


def load_audio(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load audio file, resample to target rate, return mono float32 array."""
    arr, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    return arr.astype(np.float32)


def compute_fbank_kaldi(waveform: np.ndarray,
                        sample_rate: int = 16000,
                        n_mels: int = 80,
                        frame_length: int = 25,
                        frame_shift: int = 10,
                        dither: float = 1.0) -> np.ndarray:
    """Compute fbank using kaldi_native_fbank (matches SenseVoice training).

    Args:
        waveform: 1-D float32 array, range [-1.0, 1.0]
        sample_rate: Target sample rate
        n_mels: Number of mel bins
        frame_length: Frame length in ms
        frame_shift: Frame shift in ms
        dither: Dithering factor

    Returns:
        fbank: [n_frames, n_mels] float32 array
    """
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = sample_rate
    opts.frame_opts.dither = dither
    opts.frame_opts.window_type = "hamming"
    opts.frame_opts.frame_shift_ms = float(frame_shift)
    opts.frame_opts.frame_length_ms = float(frame_length)
    opts.mel_opts.num_bins = n_mels
    opts.energy_floor = 0
    opts.frame_opts.snip_edges = True
    opts.mel_opts.debug_mel = False

    # Scale waveform to int16 range (SenseVoice convention)
    scaled_waveform = waveform * (1 << 15)

    fbank_fn = knf.OnlineFbank(opts)
    fbank_fn.accept_waveform(sample_rate, scaled_waveform.tolist())
    frames = fbank_fn.num_frames_ready
    mat = np.empty([frames, n_mels], dtype=np.float32)
    for i in range(frames):
        mat[i, :] = fbank_fn.get_frame(i)
    return mat


def apply_lfr(fbank: np.ndarray, lfr_m: int = 7, lfr_n: int = 6) -> np.ndarray:
    """Apply Low Frame Rate (LFR) stacking with left-context padding.

    Matches SenseVoice official implementation:
    - Left padding: tile the first frame (lfr_m-1)//2 times
    - Stack lfr_m consecutive frames, advance by lfr_n
    - Right pad last frame if needed

    Args:
        fbank: [n_frames, n_mels]
        lfr_m: Number of frames to stack
        lfr_n: Frame skip rate

    Returns:
        [n_out_frames, lfr_m * n_mels]
    """
    T = fbank.shape[0]
    T_lfr = int(np.ceil(T / lfr_n))
    # Left-context padding: repeat first frame
    left_padding = np.tile(fbank[0], ((lfr_m - 1) // 2, 1))
    padded = np.vstack((left_padding, fbank))
    T_padded = padded.shape[0]

    stacked = []
    for i in range(T_lfr):
        start = i * lfr_n
        if lfr_m <= T_padded - start:
            stacked.append(padded[start:start + lfr_m].reshape(1, -1))
        else:
            # Process last LFR frame with right padding
            frame = padded[start:].reshape(-1)
            num_padding = lfr_m - (T_padded - start)
            for _ in range(num_padding):
                frame = np.hstack((frame, padded[-1]))
            stacked.append(frame)

    return np.vstack(stacked).astype(np.float32)


def load_cmvn(cmvn_file: str) -> np.ndarray:
    """Load CMVN from Kaldi-format am.mvn file.

    Returns:
        cmvn: [2, dim] array where cmvn[0]=means (AddShift), cmvn[1]=vars (Rescale)
    """
    means_list = []
    vars_list = []
    with open(cmvn_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for i in range(len(lines)):
        line_item = lines[i].split()
        if line_item[0] == "<AddShift>":
            line_item = lines[i + 1].split()
            if line_item[0] == "<LearnRateCoef>":
                means_list = list(line_item[3:(len(line_item) - 1)])
        elif line_item[0] == "<Rescale>":
            line_item = lines[i + 1].split()
            if line_item[0] == "<LearnRateCoef>":
                vars_list = list(line_item[3:(len(line_item) - 1)])
    means = np.array(means_list, dtype=np.float64)
    vars_ = np.array(vars_list, dtype=np.float64)
    return np.array([means, vars_])


def apply_cmvn(features: np.ndarray, cmvn: np.ndarray) -> np.ndarray:
    """Apply CMVN: (features + means) * vars.

    Args:
        features: [n_frames, dim]
        cmvn: [2, dim] from load_cmvn

    Returns:
        Normalized features [n_frames, dim]
    """
    frame, dim = features.shape
    means = np.tile(cmvn[0:1, :dim], (frame, 1))
    vars_ = np.tile(cmvn[1:2, :dim], (frame, 1))
    return (features + means) * vars_


