"""
后端启动路径测试。

回归: 模型改为清单式管理后, ``config.yaml`` 不再有 ``model.onnx`` /
``model.tokens`` 等键, 但 ``run_server.py`` 的启动横幅仍在读
``config['model']['onnx_path']`` —— 后端一启动就 KeyError 崩溃 (退出码 1),
端口 11235 没有监听, 前端所有模型相关功能都拿不到数据。
这类缺陷单元测试抓不到 (没人 import run_server), 因此这里显式覆盖:

  1. ``load_config`` 对缺键的旧配置给出完整默认值, 不抛 KeyError
  2. ``run_server.main`` 能在真实 config.yaml 下走到 uvicorn.run (被 mock 拦截)
"""
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.core.config_loader import DEFAULTS, load_config  # noqa: E402


class TestLoadConfig:
    """配置加载必须容忍缺段缺键 (旧配置升级后不能崩)。"""

    def test_real_config_has_all_required_sections(self):
        config = load_config("config.yaml")
        for section in ("model", "audio", "inference", "server", "engines"):
            assert section in config, section

    def test_model_dir_is_absolute(self):
        config = load_config("config.yaml")
        assert Path(config["model"]["dir"]).is_absolute()

    def test_minimal_config_gets_defaults(self, tmp_path):
        """只写 server 段的最小配置也要能加载。"""
        cfg = tmp_path / "min.yaml"
        cfg.write_text("server:\n  host: '127.0.0.1'\n  port: 11235\n", encoding="utf-8")
        config = load_config(str(cfg))
        assert config["model"]["dir"]
        assert config["audio"]["sample_rate"] == DEFAULTS["audio"]["sample_rate"]
        assert config["inference"]["device"] == "cpu"

    def test_empty_config_does_not_raise(self, tmp_path):
        cfg = tmp_path / "empty.yaml"
        cfg.write_text("", encoding="utf-8")
        config = load_config(str(cfg))
        assert config["model"]["dir"]

    def test_removed_legacy_keys_are_absent(self):
        """旧的自研 ONNX 路径键必须已从配置中移除。"""
        config = load_config("config.yaml")
        for legacy in ("onnx", "tokens", "onnx_path", "tokens_path", "model_config", "config_path"):
            assert legacy not in config["model"], f"残留旧键: {legacy}"

    def test_cmvn_is_gone(self):
        """fbank/LFR/CMVN 预处理已删, cmvn_file 不应再出现。"""
        config = load_config("config.yaml")
        assert "cmvn_file" not in config.get("audio", {})

    def test_engine_keys_match_catalog_engine_names(self):
        """config.yaml 的 engines.<key> 必须与清单的 engine_name 对得上。"""
        from app.core import model_catalog

        config = load_config("config.yaml")
        configured = set(config["engines"])
        expected = {spec["engine_name"] for spec in model_catalog.CATALOG["models"]}
        assert expected <= configured, f"清单里有未配置的引擎: {expected - configured}"


class TestRunServerStartup:
    """run_server.main 必须能真正走到启动 uvicorn (回归: 曾 KeyError 崩溃)。"""

    def test_main_reaches_uvicorn(self, monkeypatch, tmp_path):
        import run_server

        calls = {}
        monkeypatch.setattr(run_server.uvicorn, "run", lambda *a, **k: calls.update(kwargs=k))
        monkeypatch.setattr(
            sys, "argv",
            ["run_server.py", "--config", str(BACKEND / "config.yaml"),
             "--log-level", "ERROR"],
        )
        run_server.main()
        assert "kwargs" in calls, "main() 未走到 uvicorn.run —— 启动路径仍然会崩"

    def test_startup_banner_reports_catalog_not_legacy_keys(self, monkeypatch, capsys):
        """启动横幅报的是清单与已安装模型, 不再引用已删除的旧键。"""
        import run_server

        monkeypatch.setattr(run_server.uvicorn, "run", lambda *a, **k: None)
        monkeypatch.setattr(
            sys, "argv",
            ["run_server.py", "--config", str(BACKEND / "config.yaml"), "--log-level", "ERROR"],
        )
        run_server.main()
        # 注意: setup_logging 会替换 builtins.print, 所以从 capsys 读真实输出
        joined = capsys.readouterr().out
        assert "Model root:" in joined
        assert "Installed models:" in joined
        assert "onnx_path" not in joined
