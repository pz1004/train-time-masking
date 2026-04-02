from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


def dumps_toml(data: Mapping[str, Any]) -> str:
    lines: list[str] = []
    _emit_table(lines, data, ())
    return ("\n".join(lines).rstrip() + "\n") if lines else ""


def _emit_table(lines: list[str], table: Mapping[str, Any], prefix: tuple[str, ...]) -> None:
    scalars: list[str] = []
    child_tables: list[tuple[str, Mapping[str, Any]]] = []
    array_tables: list[tuple[str, Sequence[Mapping[str, Any]]]] = []

    for key, value in table.items():
        if isinstance(value, Mapping):
            child_tables.append((key, value))
            continue
        if _is_array_of_tables(value):
            array_tables.append((key, value))
            continue
        scalars.append(f"{key} = {_format_value(value)}")

    if prefix:
        if lines:
            lines.append("")
        lines.append(f"[{'.'.join(prefix)}]")

    lines.extend(scalars)

    for key, value in child_tables:
        _emit_table(lines, value, prefix + (key,))

    for key, value in array_tables:
        for item in value:
            if lines:
                lines.append("")
            lines.append(f"[[{'.'.join(prefix + (key,))}]]")
            for item_key, item_value in item.items():
                if isinstance(item_value, Mapping):
                    raise TypeError("Nested tables inside array-of-tables are not supported in this serializer.")
                if _is_array_of_tables(item_value):
                    raise TypeError("Nested array-of-tables are not supported in this serializer.")
                lines.append(f"{item_key} = {_format_value(item_value)}")


def _is_array_of_tables(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and bool(value) and all(
        isinstance(item, Mapping) for item in value
    )


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value: {type(value)!r}")

