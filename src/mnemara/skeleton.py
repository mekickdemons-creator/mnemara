"""Pure-function Python skeleton extractor using stdlib ast only."""
from __future__ import annotations

import ast


def extract_python_skeleton(source: str) -> str:
    """Extract function and class signatures with docstrings from Python source.

    Keeps:
      - Top-level imports
      - Class declarations with their method signatures
      - Function signatures (with full type hints)
      - Docstrings (module, class, function)

    Strips:
      - Function/method bodies (replaced with ``...``)
      - Module-level statements that aren't imports, class defs, or function defs

    Returns "" for empty source.
    Returns "# syntax error: {e}" for unparseable source.
    """
    if not source or not source.strip():
        return ""

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"# syntax error: {e}"

    lines: list[str] = []

    # Module docstring
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        doc = tree.body[0].value.value
        lines.append(_format_docstring(doc, indent=0))
        lines.append("")

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            lines.append(ast.unparse(node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lines.extend(_function_skeleton(node, indent=0))
        elif isinstance(node, ast.ClassDef):
            lines.extend(_class_skeleton(node))
        # All other module-level statements are stripped

    return "\n".join(lines)


def _format_docstring(doc: str, indent: int) -> str:
    """Format a docstring value as a triple-quoted string with given indent."""
    prefix = " " * indent
    # If it contains newlines, use multi-line format
    if "\n" in doc:
        return f'{prefix}"""{doc}"""'
    else:
        return f'{prefix}"""{doc}"""'


def _function_skeleton(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    indent: int,
) -> list[str]:
    """Return skeleton lines for a function/method definition."""
    lines: list[str] = []
    prefix = " " * indent

    # Build signature using ast.unparse for arguments
    args_str = ast.unparse(node.args)
    returns_str = ""
    if node.returns is not None:
        returns_str = f" -> {ast.unparse(node.returns)}"

    async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    sig = f"{prefix}{async_prefix}def {node.name}({args_str}){returns_str}:"
    lines.append(sig)

    # Check for docstring
    body_indent = " " * (indent + 4)
    docstring = _get_docstring(node)
    if docstring is not None:
        lines.append(f'{body_indent}"""{docstring}"""')

    lines.append(f"{body_indent}...")
    return lines


def _class_skeleton(node: ast.ClassDef) -> list[str]:
    """Return skeleton lines for a class definition."""
    lines: list[str] = []

    # Build class declaration
    bases = [ast.unparse(b) for b in node.bases]
    keywords = [ast.unparse(k) for k in node.keywords]
    all_bases = bases + keywords
    if all_bases:
        lines.append(f"class {node.name}({', '.join(all_bases)}):")
    else:
        lines.append(f"class {node.name}:")

    # Class docstring
    docstring = _get_docstring(node)
    if docstring is not None:
        lines.append(f'    """{docstring}"""')
        lines.append("")

    has_methods = False
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method_lines = _function_skeleton(item, indent=4)
            lines.extend(method_lines)
            lines.append("")
            has_methods = True

    # If the class has no methods and no docstring, add ...
    if not has_methods and docstring is None:
        lines.append("    ...")

    # Remove trailing empty line from last method if present
    while lines and lines[-1] == "":
        lines.pop()

    return lines


def _get_docstring(node: ast.AST) -> str | None:
    """Return docstring value if the node body starts with a string literal."""
    body = getattr(node, "body", [])
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[0].value.value
    return None
