"""Tests for `mnemara role --set-from-url`."""
from __future__ import annotations

import io
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from mnemara import cli, config as config_mod, paths


@pytest.fixture
def instance(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "root", lambda: tmp_path)
    name = "urltest"
    config_mod.init_instance(name, role_doc_path="")
    return name


def _run(args):
    return CliRunner().invoke(cli.main, args)


def test_requires_one_of_set_or_url(instance):
    res = _run(["role", "--instance", instance])
    assert res.exit_code != 0
    assert "either --set or --set-from-url" in res.output


def test_mutually_exclusive(instance, tmp_path):
    p = tmp_path / "r.md"
    p.write_text("x")
    res = _run([
        "role", "--instance", instance,
        "--set", str(p),
        "--set-from-url", "https://example.com/r.md",
    ])
    assert res.exit_code != 0
    assert "mutually exclusive" in res.output


def test_rejects_non_https(instance):
    res = _run([
        "role", "--instance", instance,
        "--set-from-url", "http://example.com/r.md",
    ])
    assert res.exit_code != 0
    assert "https://" in res.output


def test_rejects_oversized(instance):
    big = b"a" * 1_000_001
    fake = io.BytesIO(big)
    fake.__enter__ = lambda s: s
    fake.__exit__ = lambda *a: None
    with patch("urllib.request.urlopen", return_value=fake):
        res = _run([
            "role", "--instance", instance,
            "--set-from-url", "https://example.com/big.md",
        ])
    assert res.exit_code != 0
    assert "too large" in res.output


def test_rejects_non_utf8(instance):
    fake = io.BytesIO(b"\xff\xfe\x00bad bytes")
    fake.__enter__ = lambda s: s
    fake.__exit__ = lambda *a: None
    with patch("urllib.request.urlopen", return_value=fake):
        res = _run([
            "role", "--instance", instance,
            "--set-from-url", "https://example.com/binary.md",
        ])
    assert res.exit_code != 0
    assert "UTF-8" in res.output


def test_downloads_and_persists(instance, tmp_path):
    body = b"# My Role\n\nDo the thing.\n"
    fake = io.BytesIO(body)
    fake.__enter__ = lambda s: s
    fake.__exit__ = lambda *a: None
    with patch("urllib.request.urlopen", return_value=fake):
        res = _run([
            "role", "--instance", instance,
            "--set-from-url", "https://example.com/role.md",
        ])
    assert res.exit_code == 0, res.output
    cfg = config_mod.load(instance)
    assert cfg.role_doc_path.endswith("role.md")
    assert (paths.instance_dir(instance) / "role.md").read_text() == body.decode()
