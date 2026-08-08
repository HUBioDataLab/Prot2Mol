import pytest
import transformers

from prot2mol.io import hf_utils


def test_resolve_model_path_returns_existing_input(tmp_path):
    existing = tmp_path / "existing_model_dir"
    existing.mkdir()
    assert hf_utils.resolve_model_path(str(existing)) == str(existing)


def test_resolve_model_path_uses_snapshot_dir(tmp_path, monkeypatch):
    models_base = tmp_path / "models"
    snapshot = models_base / "models--zjunlp--MolGen-large" / "snapshots" / "snap-1"
    snapshot.mkdir(parents=True)
    monkeypatch.setenv("MODELS_BASE_PATH", str(models_base))

    resolved = hf_utils.resolve_model_path("zjunlp--MolGen-large")
    assert resolved == str(snapshot)


def test_resolve_model_path_honors_huggingface_main_ref(tmp_path):
    models_base = tmp_path / "models"
    model_dir = models_base / "models--zjunlp--MolGen-large"
    old_snapshot = model_dir / "snapshots" / "old"
    current_snapshot = model_dir / "snapshots" / "current"
    old_snapshot.mkdir(parents=True)
    current_snapshot.mkdir()
    (model_dir / "refs").mkdir()
    (model_dir / "refs" / "main").write_text("current\n", encoding="utf-8")

    assert hf_utils.resolve_model_path(
        "zjunlp/MolGen-large",
        models_base=str(models_base),
    ) == str(current_snapshot)


def test_resolve_model_path_falls_back_to_canonical_hub_id(tmp_path):
    assert hf_utils.resolve_model_path(
        "zjunlp--MolGen-large", models_base=str(tmp_path)
    ) == "zjunlp/MolGen-large"


def test_load_molgen_tokenizer_uses_resolved_path_and_padding(monkeypatch):
    called = {}

    class DummyAutoTokenizer:
        @staticmethod
        def from_pretrained(path):
            called["path"] = path
            return type("Tokenizer", (), {"tokenizer_path": path, "padding_side": None})()

    monkeypatch.setattr(hf_utils, "resolve_model_path", lambda *a, **k: "/tmp/molgen")
    monkeypatch.setattr(transformers, "AutoTokenizer", DummyAutoTokenizer)

    tokenizer = hf_utils.load_molgen_tokenizer(padding_side="right")
    assert tokenizer.tokenizer_path == "/tmp/molgen"
    assert tokenizer.padding_side == "right"
    assert called["path"] == "/tmp/molgen"


def test_resolve_checkpoint_file_bin_then_safetensors_and_error(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        hf_utils._resolve_checkpoint_file(str(model_dir))

    safetensors_file = model_dir / "model.safetensors"
    safetensors_file.write_bytes(b"x")
    assert hf_utils._resolve_checkpoint_file(str(model_dir)) == str(safetensors_file)

    pytorch_file = model_dir / "pytorch_model.bin"
    pytorch_file.write_bytes(b"x")
    assert hf_utils._resolve_checkpoint_file(str(model_dir)) == str(pytorch_file)
    assert hf_utils._resolve_checkpoint_file(str(pytorch_file)) == str(pytorch_file)
