# efficient-vla-benchmark

Reproduce complete VLA acceleration methods on pinned native model code, with
optional backend packages and one shared control/evaluation/profiling contract.
This repository contains no survey manuscript, model weights or datasets.

## Architecture

- `src/efficient_vla_benchmark/`: lightweight contracts, method catalog and shared harnesses.
- `backends/<backend>/`: independently installed backend plugin; its `upstream/` is a pinned submodule.
- `methods/<category>/<method>/` inside the main package: one paper, with explicit per-backend implementations.
- `configs/`: portable presets; filesystem resources are supplied only on the command line.
- `third_party/`: pinned algorithm references, runtime libraries and benchmark sources.

No backend is a mandatory dependency of the main package. Install only the backend
you use. A source-only integration is not an implemented or validated backend.
One run uses the native baseline or exactly one complete acceleration method.

Read [installation](docs/installation.md), [architecture](docs/architecture.md)
and [contribution rules](CONTRIBUTING.md). Validation records will distinguish
development smoke checks, measured timing and formal success-rate campaigns.

