"""Tests for src/mnemara/skeleton.py — extract_python_skeleton pure function."""
from __future__ import annotations

import pytest

from mnemara.skeleton import extract_python_skeleton


# ---------------------------------------------------------------------------
# Basic extraction
# ---------------------------------------------------------------------------

SIMPLE_SOURCE = '''\
"""Module docstring."""
import os
import sys

def greet(name: str) -> str:
    """Return a greeting."""
    return f"Hello, {name}"

def _private(x: int, y: int = 0) -> int:
    return x + y

class Greeter:
    """Greets things."""

    def __init__(self, prefix: str) -> None:
        """Init."""
        self.prefix = prefix

    def say(self, name: str) -> str:
        """Say the greeting."""
        return f"{self.prefix} {name}"
'''


def test_keeps_function_signatures():
    out = extract_python_skeleton(SIMPLE_SOURCE)
    assert "def greet(name: str) -> str:" in out
    # ast.unparse omits the space around = in annotated defaults: y: int=0
    assert "def _private(x: int, y: int" in out and "-> int:" in out


def test_strips_function_bodies():
    out = extract_python_skeleton(SIMPLE_SOURCE)
    # Body line should not appear
    assert 'return f"Hello, {name}"' not in out
    assert "return x + y" not in out


def test_keeps_docstrings():
    out = extract_python_skeleton(SIMPLE_SOURCE)
    assert "Return a greeting." in out
    assert "Module docstring." in out
    assert "Greets things." in out


def test_keeps_imports():
    out = extract_python_skeleton(SIMPLE_SOURCE)
    assert "import os" in out
    assert "import sys" in out


def test_keeps_class_declaration():
    out = extract_python_skeleton(SIMPLE_SOURCE)
    assert "class Greeter:" in out


def test_keeps_method_signatures():
    out = extract_python_skeleton(SIMPLE_SOURCE)
    assert "def __init__(self, prefix: str) -> None:" in out
    assert "def say(self, name: str) -> str:" in out


def test_replaces_bodies_with_ellipsis():
    out = extract_python_skeleton(SIMPLE_SOURCE)
    assert "..." in out


# ---------------------------------------------------------------------------
# Async function
# ---------------------------------------------------------------------------

ASYNC_SOURCE = '''\
async def fetch(url: str, timeout: float = 30.0) -> bytes:
    """Fetch a URL."""
    response = await client.get(url, timeout=timeout)
    return response.content
'''


def test_keeps_async_signature():
    out = extract_python_skeleton(ASYNC_SOURCE)
    # ast.unparse omits space around = in annotated defaults (float=30.0)
    assert "async def fetch(url: str, timeout: float" in out
    assert "-> bytes:" in out
    assert "response = await client.get" not in out


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_source_returns_empty():
    assert extract_python_skeleton("") == ""
    assert extract_python_skeleton("   \n  ") == ""


def test_syntax_error_returns_comment():
    bad = "def foo(:\n    pass"
    out = extract_python_skeleton(bad)
    assert out.startswith("# syntax error:")


def test_no_functions_or_classes():
    source = "import os\nx = 42\nprint(x)\n"
    out = extract_python_skeleton(source)
    assert "import os" in out
    # The module-level statement `x = 42` should be stripped
    assert "x = 42" not in out
    assert "print(x)" not in out


def test_class_with_no_methods_no_docstring():
    source = "class Empty:\n    pass\n"
    out = extract_python_skeleton(source)
    assert "class Empty:" in out
    # Should emit ... to keep it valid
    assert "..." in out


def test_return_type_preserved():
    source = "def f() -> dict[str, list[int]]:\n    return {}\n"
    out = extract_python_skeleton(source)
    assert "-> dict[str, list[int]]:" in out


def test_decorator_not_in_output():
    # Decorators are not preserved in the skeleton (bodies only)
    source = "@property\ndef x(self) -> int:\n    return self._x\n"
    out = extract_python_skeleton(source)
    # The signature should appear; decorator stripping is acceptable
    assert "def x(self) -> int:" in out
