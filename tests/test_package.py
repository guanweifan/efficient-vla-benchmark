import subprocess
import sys


def test_import_is_lightweight():
    code = (
        "import sys; import efficient_vla_benchmark; "
        "heavy = ('torch', 'transformers', 'tensorflow', 'libero'); "
        "assert not any(n in sys.modules for n in heavy)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
