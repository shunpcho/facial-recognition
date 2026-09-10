# Agent Instructions (Copilot/AI assistants)

## Instructions

- Do not preserve backward compatibility. Remove obsolete paths instead of adding compatibility layers, fallbacks, or migrations.
- Choose the simplest implementation that fully meets the current requirements. Avoid speculative abstractions, configuration, and indirection.
- Grow the system in layers. Start from the smallest version that works end to end, and add each new capability on top of a proven foundation.
- Keep components modular and concerns clearly separated.
- Prefer established, well-maintained libraries when they reduce overall complexity or improve reliability. Do not reimplement common functionality without a clear reason.
- Lean on the dependencies already in the project before writing your own implementation or adding packages. Do not assume a library lacks a capability without checking its documentation and types.
- Make architectural decisions for the long term. Do not accept a stopgap that only works for now and is meant to be replaced later.

## Overview

This document provides guidelines, best practices, and conventions for Python projects that use AI assistants. It defines agent instructions, coding standards, project structure, and dependency-management practices.

## Documentation

- Follow `.github/copilot-instructions.md` for Python coding rules.
- Consult the design documents and decisions in the `docs/` directory.

## Repository structure

The project root must contain:

- `pyproject.toml` (required; single source of truth)
- `src/<package_name>/` (package code; keep the `src/` layout consistent)
- `tests/`
- `docs/`

## Dependency management

Dependency management must use **uv**. Prefer `uv run ...` when executing tools in the managed environment.

## Key resources

Refer to the following resources for style, linting, type checking, and dependency management:

- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) - Referenced for style conventions
- [Ruff](https://docs.astral.sh/ruff/) - Fast Python linter and formatter
- [Pyright](https://github.com/microsoft/pyright) - Static type checker
- [UV](https://github.com/astral-sh/uv) - Python package manager
