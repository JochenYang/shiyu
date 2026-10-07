"""
sherpa 引擎与清单的绑定测试 (不加载真实模型)。

重点回归: ``byte_bpe`` 必须由清单的 ``engine`` 字段决定, 而不是靠子类属性。
曾经注册中心实例化的是基类 ``SherpaOfflineEngine``, 于是 ``byte_bpe`` 退回
``False``, zipformer 的 BBPE 解码静默失败 —— 逐字符时间全部拿不到, 28 秒的
长段无法断句, 整段字幕只能挤成一行。
"""
import pytest

from app.core import model_catalog
from app.core.sherpa_engine import (
    FireRedEngine,
    SenseVoiceEngine,
    SherpaOfflineEngine,
    ZipformerEngine,
)


def _spec(engine: str) -> dict:
    return {
        "id": "t", "name": "T", "engine": engine, "engine_name": "t_local",
        "directory": "t", "repository": "r", "revision": "0" * 40,
        "files": {"model.int8.onnx": 1, "tokens.txt": 1},
        "sha256": {"model.int8.onnx": "0" * 64, "tokens.txt": "0" * 64},
    }


class TestByteBpeBinding:
    """byte_bpe 由清单决定, 与实例化的是哪个类无关。"""

    @pytest.mark.parametrize("engine,expected", [
        ("zipformer", True),
        ("sensevoice", False),
        ("fire_red_asr_ctc", False),
    ])
    def test_byte_bpe_follows_catalog_engine(self, monkeypatch, engine, expected):
        monkeypatch.setitem(model_catalog.CATALOG, "models", [_spec(engine)])
        for cls in (SherpaOfflineEngine, ZipformerEngine, SenseVoiceEngine, FireRedEngine):
            instance = cls(model_id="t")
            assert instance.byte_bpe is expected, cls.__name__

    def test_zipformer_is_byte_bpe_even_when_base_class_is_instantiated(self, monkeypatch):
        """回归: 注册中心实例化基类时也必须拿到 byte_bpe=True。"""
        monkeypatch.setitem(model_catalog.CATALOG, "models", [_spec("zipformer")])
        assert SherpaOfflineEngine(model_id="t").byte_bpe is True

    def test_engine_name_comes_from_catalog(self, monkeypatch):
        """引擎标识符来自清单 engine_name, 不硬编码在类里。"""
        spec = _spec("zipformer")
        spec["engine_name"] = "custom_local"
        monkeypatch.setitem(model_catalog.CATALOG, "models", [spec])
        assert SherpaOfflineEngine(model_id="t").name == "custom_local"


class TestFactoryMapping:
    """清单 engine 字段 → sherpa-onnx 工厂方法。"""

    @pytest.mark.parametrize("engine,factory", [
        ("zipformer", "from_zipformer_ctc"),
        ("sensevoice", "from_sense_voice"),
        ("fire_red_asr_ctc", "from_fire_red_asr_ctc"),
        ("qwen3_asr", "from_qwen3_asr"),
    ])
    def test_engine_maps_to_factory(self, monkeypatch, engine, factory):
        monkeypatch.setitem(model_catalog.CATALOG, "models", [_spec(engine)])
        assert SherpaOfflineEngine(model_id="t")._factory_name() == factory

    def test_unknown_engine_is_rejected(self, monkeypatch):
        monkeypatch.setitem(model_catalog.CATALOG, "models", [_spec("no_such_engine")])
        with pytest.raises(ValueError, match="未知的引擎类型"):
            SherpaOfflineEngine(model_id="t")._factory_name()

class TestMultiFileModels:
    """Qwen3-ASR 是多文件模型, 参数组装不能写死单文件路径。

    （Whisper 不在清单中: 其 sherpa-onnx 导出的 decoder 缺 cross-attention,
    拿不到逐 token 时间戳, 接入只会让字幕变差。）
    """

    QWEN_FILES = {
        "conv_frontend.onnx": "/m/conv_frontend.onnx",
        "encoder.int8.onnx": "/m/encoder.int8.onnx",
        "decoder.int8.onnx": "/m/decoder.int8.onnx",
        "tokenizer/vocab.json": "/m/tokenizer/vocab.json",
        "tokenizer/merges.txt": "/m/tokenizer/merges.txt",
    }

    @pytest.fixture(autouse=True)
    def _catalog(self, monkeypatch):
        """_factory_args 是实例方法, 构造时会查清单, 故先注入一个占位模型。"""
        monkeypatch.setitem(model_catalog.CATALOG, "models", [_spec("qwen3_asr")])


    def test_qwen_args_use_conv_frontend_and_tokenizer_dir(self):
        eng = SherpaOfflineEngine(model_id="t")
        args = eng._factory_args(self.QWEN_FILES, "from_qwen3_asr")
        assert args["conv_frontend"].endswith("conv_frontend.onnx")
        assert args["encoder"].endswith("encoder.int8.onnx")
        assert args["decoder"].endswith("decoder.int8.onnx")
        # tokenizer 要的是目录, 不是某个文件
        assert args["tokenizer"].replace("\\", "/").endswith("/tokenizer")

    def test_pick_rejects_ambiguous_match(self):
        eng = SherpaOfflineEngine(model_id="t")
        files = {"a-encoder.onnx": "/x", "b-encoder.onnx": "/y"}
        with pytest.raises(ValueError, match="应恰为 1 个"):
            eng._pick(files, "encoder")

    def test_pick_rejects_missing_match(self):
        eng = SherpaOfflineEngine(model_id="t")
        with pytest.raises(ValueError, match="应恰为 1 个"):
            eng._pick({"only.onnx": "/x"}, "encoder")


class TestTokenToCharTimes:
    """token→逐字符时间映射。"""

    def test_whisper_space_marker_is_normalized(self):
        """Whisper 用 U+2581 表示词首空格, 归一后才能与解码文本对齐。"""
        from app.core.sherpa_engine import _char_times_from_tokens

        text = "你好 world"
        tokens = ["\u2581你", "好", "\u2581world"]
        times = [0.0, 0.5, 1.0]
        result = _char_times_from_tokens(text, tokens, times, byte_bpe=False)
        assert result is not None
        assert len(result) == len(text)
        assert result[0] == 0.0
        assert result[-1] == 1.0

    def test_mismatched_tokens_return_none_not_guess(self):
        """对不上就返回 None, 绝不用比例插值伪造时间。"""
        from app.core.sherpa_engine import _char_times_from_tokens

        assert _char_times_from_tokens("完全不同的文本", ["a"], [0.0], byte_bpe=False) is None
    def test_length_mismatch_between_tokens_and_times(self):
        from app.core.sherpa_engine import _char_times_from_tokens

        assert _char_times_from_tokens("你好", ["你", "好"], [0.0], byte_bpe=False) is None


class TestAvailability:
    """可用性只读清单, 不触发下载。"""

    def test_unavailable_when_model_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SHIYU_MODEL_ROOT", str(tmp_path / "models"))
        monkeypatch.setitem(model_catalog.CATALOG, "models", [_spec("zipformer")])
        available, reason = SherpaOfflineEngine(model_id="t").is_available()
        assert available is False
        assert "未安装" in reason

    def test_available_when_files_match(self, tmp_path, monkeypatch):
        root = tmp_path / "models"
        directory = root / "t"
        directory.mkdir(parents=True)
        (directory / "model.int8.onnx").write_bytes(b"x")
        (directory / "tokens.txt").write_bytes(b"y")
        monkeypatch.setenv("SHIYU_MODEL_ROOT", str(root))
        monkeypatch.setitem(model_catalog.CATALOG, "models", [_spec("zipformer")])
        available, reason = SherpaOfflineEngine(model_id="t").is_available()
        assert available is True, reason
