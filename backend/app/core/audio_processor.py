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
import tempfile
import numpy as np
import librosa
import kaldi_native_fbank as knf


def extract_audio_from_video(video_path: str, output_wav: str = None, sample_rate: int = 16000) -> str:
    """Extract audio from video using ffmpeg. Returns path to wav file."""
    if output_wav is None:
        output_wav = tempfile.mktemp(suffix=".wav")
    
    # Try to find ffmpeg in PATH or via environment variable
    ffmpeg_exe = os.environ.get("FFMPEG_PATH", "ffmpeg")
    
    cmd = [
        ffmpeg_exe, "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ac", "1", "-ar", str(sample_rate),
        output_wav
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        # Fallback for common WinGet path if not in PATH
        winget_ffmpeg = r"C:\Users\Administrator\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"
        if os.path.exists(winget_ffmpeg):
            cmd[0] = winget_ffmpeg
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            raise
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


def preprocess_audio(audio_path: str, is_video: bool = False, config: dict = None) -> np.ndarray:
    """Full preprocessing pipeline: extract -> fbank -> LFR -> CMVN.

    Returns:
        features: [1, n_frames, feature_dim] float32, ready for ONNX
    """
    cfg = config or {}
    sample_rate = cfg.get("sample_rate", 16000)
    n_mels = cfg.get("n_mels", 80)
    frame_length = cfg.get("frame_length", 25)
    frame_shift = cfg.get("frame_shift", 10)
    lfr_m = cfg.get("lfr_m", 7)
    lfr_n = cfg.get("lfr_n", 6)
    cmvn_file = cfg.get("cmvn_file", None)

    if is_video:
        audio_path = extract_audio_from_video(audio_path, sample_rate=sample_rate)

    waveform = load_audio(audio_path, sample_rate)
    fbank = compute_fbank_kaldi(waveform, sample_rate, n_mels, frame_length, frame_shift)
    features = apply_lfr(fbank, lfr_m, lfr_n)

    # Apply CMVN if available
    if cmvn_file and os.path.exists(cmvn_file):
        cmvn = load_cmvn(cmvn_file)
        features = apply_cmvn(features, cmvn).astype(np.float32)

    # Add batch dimension: [1, n_frames, feature_dim]
    return np.expand_dims(features, axis=0)
