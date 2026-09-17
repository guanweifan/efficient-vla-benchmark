# External method template

This separately installable fixture scales an example backend's actions to test
discovery, compatibility and detach semantics. It is not an acceleration algorithm.
Copy its package structure, not its mechanism; implement the complete paper method
and replace the metadata, config and factory. See docs/adding_method.md.

```bash
python -m pip install -e templates/method
evb list
```
