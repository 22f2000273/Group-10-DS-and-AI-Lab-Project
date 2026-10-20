# Contributing Guidelines

## Coding Conventions
- Follow PEP 8 for Python code.
- Use clear naming and modular functions.
- Keep reusable logic in `src/` and deployment UI logic in `app/`.

## Branch Naming
- Feature branches: `feature/<short-description>`
- Bugfix branches: `fix/<short-description>`
- Documentation branches: `docs/<short-description>`

## Pull Request Process
1. Create a focused branch from the latest default branch.
2. Keep PRs small and scoped to one logical change.
3. Include a clear summary, testing notes, and screenshots (if UI changes).
4. Request review from at least one teammate.

## Team Responsibilities
- Keep milestone artifacts updated in `docs/`.
- Ensure experiments are reproducible via `configs/` and `src/`.
- Add or update tests in `tests/` for new functionality.
