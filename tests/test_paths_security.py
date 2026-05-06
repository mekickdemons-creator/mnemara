"""Regression tests for instance-name validation in paths.py."""
from __future__ import annotations

import pytest

from mnemara import paths


def test_valid_instance_names():
    for name in ["a", "default", "my-instance", "test_1", "v1.2", "A.B-C_d"]:
        assert paths.instance_dir(name).name == name


def test_rejects_path_traversal():
    for name in ["../foo", "../../etc", "a/b", "a\\b"]:
        with pytest.raises(ValueError):
            paths.instance_dir(name)


def test_rejects_absolute_and_special():
    for name in ["/etc", "", ".", "..", ".hidden", "-leading-dash"]:
        with pytest.raises(ValueError):
            paths.instance_dir(name)


def test_rejects_overlong_name():
    with pytest.raises(ValueError):
        paths.instance_dir("a" * 65)
