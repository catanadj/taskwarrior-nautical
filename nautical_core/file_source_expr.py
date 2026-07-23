from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass

from . import file_resource_limits as resource_limits


@dataclass(frozen=True)
class FileSource:
    pattern: str
    modifier_layers: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedFileSource:
    path: str
    display_name: str
    modifier_layers: tuple[str, ...] = ()


@dataclass(frozen=True)
class FileSourceResolution:
    sources: tuple[ResolvedFileSource, ...]
    unmatched_patterns: tuple[str, ...]


class _ExpressionParser:
    def __init__(self, value: str, *, label: str) -> None:
        self.value = value
        self.label = label
        self.pos = 0

    def parse(self) -> tuple[FileSource, ...]:
        self._skip_space()
        if self.pos >= len(self.value):
            return ()
        sources = self._parse_expression(stop=None)
        self._skip_space()
        if self.pos != len(self.value):
            raise self._error(f"unexpected '{self.value[self.pos]}'")
        return tuple(sources)

    def _parse_expression(self, *, stop: str | None) -> list[FileSource]:
        sources = self._parse_branch(stop=stop)
        while True:
            self._skip_space()
            if stop is not None and self._peek() == stop:
                return sources
            if self._peek() != "|":
                if stop is not None and self.pos >= len(self.value):
                    raise self._error(f"missing closing '{stop}'")
                return sources
            self.pos += 1
            self._skip_space()
            if self.pos >= len(self.value) or self._peek() in {"|", ")"}:
                raise self._error("empty branch")
            sources.extend(self._parse_branch(stop=stop))

    def _parse_branch(self, *, stop: str | None) -> list[FileSource]:
        self._skip_space()
        if self._peek() == "(":
            self.pos += 1
            self._skip_space()
            if self._peek() == ")":
                raise self._error("empty group")
            sources = self._parse_expression(stop=")")
            if self._peek() != ")":
                raise self._error("missing closing ')'")
            self.pos += 1
            suffix = self._read_until_delimiter(stop=stop).strip()
            if suffix and not suffix.startswith("@"):
                raise self._error("group modifiers must start with '@'")
            if suffix:
                sources = [
                    FileSource(source.pattern, source.modifier_layers + (suffix,))
                    for source in sources
                ]
            return sources

        raw = self._read_until_delimiter(stop=stop).strip()
        if not raw:
            raise self._error("empty branch")
        pattern, marker, modifier_text = raw.partition("@")
        pattern = validate_file_source_pattern(pattern.strip(), label=self.label)
        modifiers = (f"@{modifier_text.strip()}",) if marker else ()
        return [FileSource(pattern, modifiers)]

    def _read_until_delimiter(self, *, stop: str | None) -> str:
        start = self.pos
        while self.pos < len(self.value):
            char = self.value[self.pos]
            if char == "|" or char == ")" or (stop is not None and char == stop):
                break
            if char == "(":
                raise self._error("'(' is only allowed at the start of a branch")
            self.pos += 1
        return self.value[start:self.pos]

    def _skip_space(self) -> None:
        while self.pos < len(self.value) and self.value[self.pos].isspace():
            self.pos += 1

    def _peek(self) -> str:
        return self.value[self.pos] if self.pos < len(self.value) else ""

    def _error(self, detail: str) -> ValueError:
        return ValueError(f"Invalid {self.label} expression at character {self.pos + 1}: {detail}.")


def validate_file_source_pattern(value: str | None, *, label: str) -> str:
    pattern = str(value or "").strip()
    if not pattern:
        raise ValueError(f"{label} expression contains an empty file name or pattern.")
    if (
        pattern in {".", ".."}
        or "/" in pattern
        or "\\" in pattern
        or os.path.isabs(pattern)
        or "\n" in pattern
        or "\r" in pattern
    ):
        raise ValueError(f"{label} must be a file name, not a path.")
    if "[" in pattern or "]" in pattern:
        raise ValueError(f"{label} patterns support only '*' and '?' wildcards.")
    if "**" in pattern:
        raise ValueError(f"{label} does not support recursive '**' patterns.")
    return pattern


def parse_file_source_expression(value: str | None, *, label: str) -> tuple[FileSource, ...]:
    return _ExpressionParser(str(value or "").strip(), label=label).parse()


def _is_pattern(value: str) -> bool:
    return "*" in value or "?" in value


def _resolved_safe_file(path: str, *, root: str, label: str, display_name: str) -> str:
    resolved = os.path.realpath(path)
    try:
        contained = os.path.commonpath((root, resolved)) == root
    except ValueError:
        contained = False
    if not contained:
        raise ValueError(f"{label} '{display_name}' resolves outside its configured directory.")
    if not os.path.isfile(resolved):
        return ""
    return resolved


def _configured_root(directory: str | None, *, label: str) -> str:
    raw = str(directory or "").strip()
    if not raw:
        raise ValueError(f"{label}_dir is not configured.")
    root = os.path.realpath(os.path.abspath(os.path.expanduser(raw)))
    if not os.path.isdir(root):
        raise ValueError(f"{label}_dir does not exist or is not a directory.")
    return root


def resolve_file_source_expression(
    value: str | None,
    directory: str | None,
    *,
    label: str,
) -> FileSourceResolution:
    parsed = parse_file_source_expression(value, label=label)
    return resolve_file_sources(parsed, directory, label=label)


def resolve_file_sources(
    parsed: tuple[FileSource, ...],
    directory: str | None,
    *,
    label: str,
) -> FileSourceResolution:
    if not parsed:
        return FileSourceResolution((), ())
    root = _configured_root(directory, label=label)
    resolved_sources: list[ResolvedFileSource] = []
    unmatched: list[str] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    directory_files: list[str] = []
    if any(_is_pattern(source.pattern) for source in parsed):
        scanned = 0
        with os.scandir(root) as entries:
            for entry in entries:
                scanned += 1
                if scanned > resource_limits.MAX_DIRECTORY_ENTRIES:
                    raise ValueError(
                        f"{label}_dir contains more than "
                        f"{resource_limits.MAX_DIRECTORY_ENTRIES} entries; use a smaller source directory."
                    )
                if entry.name.startswith("."):
                    continue
                try:
                    if entry.is_file(follow_symlinks=True):
                        directory_files.append(entry.name)
                except OSError:
                    continue
        directory_files.sort()

    for source in parsed:
        names: list[str]
        if _is_pattern(source.pattern):
            match_pattern = "*" if source.pattern in {"*", "*.*"} else source.pattern
            names = [name for name in directory_files if fnmatch.fnmatchcase(name, match_pattern)]
            if not names:
                unmatched.append(source.pattern)
                continue
        else:
            names = [source.pattern]

        matched_any = False
        for name in names:
            candidate = os.path.join(root, name)
            path = _resolved_safe_file(candidate, root=root, label=label, display_name=name)
            if not path:
                if not _is_pattern(source.pattern):
                    raise ValueError(f"{label} '{name}' was not found in {label}_dir.")
                continue
            matched_any = True
            key = (path, source.modifier_layers)
            if key in seen:
                continue
            seen.add(key)
            if len(resolved_sources) >= resource_limits.MAX_RESOLVED_FILES:
                raise ValueError(
                    f"{label} resolves to more than "
                    f"{resource_limits.MAX_RESOLVED_FILES} files; narrow the expression."
                )
            resolved_sources.append(
                ResolvedFileSource(
                    path=path,
                    display_name=name,
                    modifier_layers=source.modifier_layers,
                )
            )
        if _is_pattern(source.pattern) and not matched_any:
            unmatched.append(source.pattern)

    return FileSourceResolution(tuple(resolved_sources), tuple(dict.fromkeys(unmatched)))
