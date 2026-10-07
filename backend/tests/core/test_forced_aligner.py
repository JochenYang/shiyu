"""
强制对齐器单元测试 (不加载真实模型)。

覆盖: 词切分、LIS 修正、pad 数公式、mel 特征与官方逐元素一致。
ONNX 推理本身的端到端验证见 repo/progress/handoffs 交接文档
(需 1GB 模型, 不适合放进单测)。
"""
import numpy as np
import pytest

from app.core import forced_aligner as fa


class TestSplitWords:
    """官方 tokenize_space_lang: CJK 逐字, 拉丁连段成词。"""

    def test_pure_cjk_is_per_char(self):
        assert fa.split_words("你好世界") == ["你", "好", "世", "界"]

    def test_latin_groups_into_words(self):
        assert fa.split_words("hello world") == ["hello", "world"]

    def test_mixed_text(self):
        # CJK 逐字; 拉丁与数字连续段成词; '+' 等符号被丢弃。
        # 注意这是对齐粒度, 不是断句粒度 —— 对齐器只保证每个词
        # 在音频里有落点, 合词交给 caption_builder。
        assert fa.split_words("用C++写5G") == ["用", "C", "写", "5G"]

    def test_punctuation_only_separates_never_merges(self):
        # 逗号只是分隔符: 两侧 CJK 仍然逐字, 不会被拼成词
        assert fa.split_words("你好，世界") == ["你", "好", "世", "界"]
    def test_empty_and_whitespace(self):
        assert fa.split_words("") == []
        assert fa.split_words("   ") == []

    def test_apostrophe_is_kept(self):
        assert fa.split_words("it's fine") == ["it's", "fine"]


class TestCharTimesFor:
    """词级时间 → 逐字符时间 (caption_builder 的契约)。"""

    @staticmethod
    def _aligner_with(items):
        """造一个只回放预设对齐结果的假对齐器 (不碰 onnxruntime)。"""
        from app.core.forced_aligner import ForcedAligner

        eng = ForcedAligner.__new__(ForcedAligner)
        eng._session = object()
        eng._tokenizer = object()
        eng._loaded_key = ("k",)
        eng._preset = items

        def fake_align(self, audio, text, sr=16000):
            return self._preset

        eng.align = fake_align.__get__(eng)
        return eng

    def test_per_char_times_inherit_word_start(self):
        items = [
            {"text": "你", "start": 0.3, "end": 0.4},
            {"text": "好", "start": 0.4, "end": 0.5},
        ]
        eng = self._aligner_with(items)
        out = eng.char_times_for(np.zeros(16000, dtype=np.float32), "你好")
        assert out == [0.3, 0.4]

    def test_spaces_inherit_previous_word(self):
        items = [{"text": "hi", "start": 0.2, "end": 0.5}]
        eng = self._aligner_with(items)
        out = eng.char_times_for(np.zeros(16000, dtype=np.float32), "hi ")
        # 拖尾空格继承前面词的时间; 文本须被对齐结果完整覆盖
        assert len(out) == 3
        assert out == [0.2, 0.2, 0.2]

    def test_uncovered_char_returns_none(self):
        """文本里有对齐结果覆盖不到的字符 → None, 不编时间。"""
        items = [{"text": "你", "start": 0.3, "end": 0.4}]
        eng = self._aligner_with(items)
        assert eng.char_times_for(np.zeros(16000, dtype=np.float32), "你好") is None

    def test_align_failure_returns_none(self):
        from unittest.mock import patch

        from app.core import forced_aligner

        eng = forced_aligner.ForcedAligner.__new__(forced_aligner.ForcedAligner)
        # patch.object 拦截 char_times_for 内部的 self.align 调用,
        # 比手工赋绑定函数可靠
        with patch.object(forced_aligner.ForcedAligner, "align",
                          side_effect=ValueError("输出时间戳不足")):
            assert eng.char_times_for(np.zeros(8, dtype=np.float32), "你好") is None


class TestGetFeatExtractOutputLengths:
    """官方长度公式: 帧数 → 音频 token 数。"""

    def test_known_value(self):
        # 3000 帧 → 390 (与实测 ScatterND 要求一致)
        assert fa._get_feat_extract_output_lengths(3000) == 390

    def test_monotonic(self):
        lengths = [fa._get_feat_extract_output_lengths(n) for n in (562, 1000, 3000)]
        assert lengths == sorted(lengths)


class TestFixTimestamps:
    """LIS 修正: 锚点保留, 乱序段内插或取邻近锚, 个数不变。"""

    def test_already_sorted_unchanged(self):
        data = [1, 2, 3, 4, 5]
        assert fa._fix_timestamps(data) == data

    def test_single_outlier_becomes_neighbor(self):
        data = [1, 2, 50, 3, 4]
        fixed = fa._fix_timestamps(data)
        assert len(fixed) == len(data)
        assert fixed == sorted(fixed)

    def test_count_is_preserved(self):
        data = [10, 1, 20, 2, 30, 3]
        assert len(fa._fix_timestamps(data)) == len(data)

    def test_empty(self):
        assert fa._fix_timestamps([]) == []

    def test_all_equal(self):
        assert fa._fix_timestamps([5, 5, 5]) == [5, 5, 5]


class TestLogMel:
    """mel 特征须与 transformers.WhisperFeatureExtractor 一致。"""

    def test_shape_is_fixed_128x3000(self):
        audio = np.zeros(16000 * 3, dtype=np.float32)
        out = fa.log_mel_spectrogram(audio)
        assert out.shape == (128, 3000)
        assert out.dtype == np.float32

    def test_short_audio_is_padded_to_30s(self):
        out = fa.log_mel_spectrogram(np.zeros(8000, dtype=np.float32))
        assert out.shape == (128, 3000)

    def test_long_audio_is_truncated_to_30s(self):
        out = fa.log_mel_spectrogram(np.zeros(16000 * 60, dtype=np.float32))
        assert out.shape == (128, 3000)

    def test_tone_produces_energy_in_expected_range(self):
        sr = 16000
        t = np.arange(sr * 2) / sr
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        out = fa.log_mel_spectrogram(audio, sr)
        # 归一化后范围应是 [-1, 1.5] 内, 且不是全零
        assert out.max() > 0.5
        assert out.min() >= -1.0

    def test_matches_official_extractor(self):
        """官方 WhisperFeatureExtractor 逐元素一致 (max diff = 0)。

        需要 transformers; 未安装时跳过 (生产代码不依赖它,
        这条测试只在开发机上核对复刻是否仍然成立)。
        """
        pytest.importorskip("transformers")
        from transformers import WhisperFeatureExtractor

        sr = 16000
        t = np.arange(sr) / sr
        audio = (0.4 * np.sin(2 * np.pi * 600 * t)).astype(np.float32)

        fe = WhisperFeatureExtractor(
            feature_size=128, sampling_rate=sr, hop_length=160,
            n_fft=400, chunk_length=30, padding_value=0.0,
            return_attention_mask=True,
        )
        ref = fe(audio, sampling_rate=sr, return_tensors="np",
                 padding="max_length", max_length=480000,
                 truncation=True)["input_features"].astype(np.float32)

        mine = fa.log_mel_spectrogram(audio, sr)
        assert mine.shape == ref.shape[1:]
        assert np.abs(mine - ref[0]).max() == pytest.approx(0.0, abs=1e-6)
