# Repository Guidelines

## Project Structure & Module Organization
- `modules/` holds the teaching modules (e.g., `modules/01-foundation/01-normalization/` with `teaching.md`, `code_guide.md`, `quiz.md`, and `experiments/`).
- `docs/`, `index.md`, and `en/` contain VitePress content; site config lives in `.vitepress/`.
- `learning_materials/` stores runnable example scripts; `docs/` and top-level `learning_log.md`/`knowledge_base.md` store notes.
- `model/`, `trainer/`, and `dataset/` mirror the MiniMind code, training scripts, and datasets used by experiments.
- `public/` and `images/` host static assets used by the docs site.

## Build, Test, and Development Commands
- Use `pnpm` for package management in this repo.
- `pnpm install` installs VitePress dependencies.
- `pnpm run docs:dev` starts the local VitePress dev server.
- `pnpm run docs:build` builds the static site to `.vitepress/dist`.
- `pnpm run docs:preview` serves the built site for verification.
- `python modules/01-foundation/01-normalization/experiments/exp1_gradient_vanishing.py` runs a sample experiment.
- `pip install -r requirements.txt` installs Python dependencies for experiments and training scripts.

## Coding Style & Naming Conventions
- Python follows PEP 8; `black` is acceptable for formatting when needed.
- Experiments should be self-contained, use fixed seeds, and include clear (often Chinese) comments.
- Experiment filenames use `exp{N}_{description}.py` and store expected outputs in `results/`.
- Module folders use `NN-topic-name` numbering for ordering (e.g., `01-normalization`).
- Markdown should use clear headings and short, instructional paragraphs.

## Testing Guidelines
- There is no formal test runner; validate changes by building docs and running relevant experiments.
- Keep experiment outputs reproducible and commit expected results under the module `results/` folder.

## Commit & Pull Request Guidelines
- Use Conventional Commit-style messages: `feat:`, `fix:`, `docs:`, `refactor:`, `perf:`, `style:`, `test:`, `chore:`, `experiment:` with optional scopes (e.g., `fix(build): ...`).
- Keep PRs small and focused on a single feature or fix.
- Use `.github/PULL_REQUEST_TEMPLATE.md` and link related issues (e.g., `Closes #123`).
- Update relevant docs when behavior or experiments change, and ensure new experiments run independently.

## Automation Notes
- AI/agent guidance is in `CLAUDE.md`; note update expectations live in `NOTE_UPDATE_GUIDE.md`.
