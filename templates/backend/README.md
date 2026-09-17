# Optional backend template

This installable CPU contract fixture demonstrates an independently distributed
backend. It is not a robot model and its timings are not benchmark evidence.
It intentionally has a 32-step chunk and a three-dimensional action space to
test that the core does not hard-code OFT geometry. LIBERO needs seven-dimensional
actions; replace the fixture with a real compatible model before LIBERO use.

Copy this directory into a new project, rename the distribution, import package and
entry-point ID, then implement your native model. No main-package registry edit is
needed. Read docs/adding_backend.md first. Never publish the example as model support.

```bash
python -m pip install -e templates/backend
evb list
```
