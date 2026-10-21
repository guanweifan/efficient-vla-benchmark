"""Render the README figure from the public ledger, never from new experiments.

Run from an installed checkout; rendering additionally requires matplotlib.
The selection, resolved presets and method metadata define coverage/categories.
The ledger supplies measured speedups (already paired) and Final-2000 outcomes.
"""

import argparse
import io
import math
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from efficient_vla_benchmark.core.config import ExperimentConfig

BACKENDS = {"openvla_oft": "OpenVLA-OFT", "cogact": "CogACT"}
# Colors and shapes are shared across panels; references are separate from methods.
STYLES = {
    "token_reduction": ("Token reduction", "#0072B2", "o"),
    "caching": ("Caching", "#009E73", "s"),
    "model_routing": ("Model routing", "#CC79A7", "^"),
    "action_generation": ("Action generation", "#D55E00", "v"),
    "action_chunking": ("Action chunking", "#B98200", "D"),
    "quantization": ("Quantization", "#7153AD", "P"),
    "model_compression": ("Model compression", "#4A8491", "X"),
}


@dataclass(frozen=True)
class Point:
    backend: str
    preset: str
    category: str
    speedup: float
    success: float
    successes: int
    supplementary: bool = False

    @property
    def key(self):
        return self.backend, self.preset


def selection_catalog(root):
    """Return primary config -> metadata without importing a model runtime."""
    metadata = {}
    methods = root / "src/efficient_vla_benchmark/methods"
    for path in sorted(methods.rglob("*.yaml")):
        if path.name != "metadata.yaml" and path.parent.name != "components":
            continue
        spec = yaml.safe_load(path.read_text())
        if spec["name"] in metadata:
            raise ValueError(f"Duplicate method metadata: {spec['name']}")
        owner = path.parent.parent if path.parent.name == "components" else path.parent
        metadata[spec["name"]] = spec, owner / "README.md"
    selection = yaml.safe_load((root / "configs/evaluation/primary_selection.yaml").read_text())
    catalog = {}
    for row in selection["configurations"]:
        config_path = Path(row["config"])
        config = ExperimentConfig.load(root / config_path)
        key = config.backend, config_path.stem
        if row["status"] != "primary" or key in catalog or config.backend not in BACKENDS:
            raise ValueError(f"Invalid primary selection: {key}")
        if config.method == "none":
            if config_path.stem != "baseline":
                raise ValueError("Native controls must not enter the primary selection")
            spec, readme = {"category": "baseline"}, None
        else:
            spec, readme = metadata[config.method]
            if config.backend not in spec["implementations"] or spec["category"] not in STYLES:
                raise ValueError(f"Undeclared support/category: {key}")
        catalog[key] = {"path": config_path, "spec": spec, "readme": readme}
    return catalog


def measured_point(backend, preset, category, speedup_cell, success_cell, supplementary=False):
    speed = re.fullmatch(r"(\d+(?:\.\d+)?)×", speedup_cell)
    result = re.fullmatch(r"(\d+(?:\.\d+)?)% \((\d+)\)", success_cell)
    if not speed or not result:
        raise ValueError(f"Missing/malformed measurement: {backend}/{preset}")
    speedup, success, count = float(speed[1]), float(result[1]), int(result[2])
    if (
        not math.isfinite(speedup)
        or speedup <= 0
        or not 0 <= count <= 2000
        or abs(success - count / 20) > 0.0051
    ):
        raise ValueError(f"Invalid Final-2000 count/rate or speedup: {backend}/{preset}")
    return Point(backend, preset, category, speedup, success, count, supplementary)


def load_points(root):
    catalog = selection_catalog(root)
    points = {}
    backend = None
    h16 = None
    for line in (root / "docs/benchmark_results.md").read_text().splitlines():
        if line.startswith("## "):
            heading = line.removeprefix("## ")
            backend = heading if heading in BACKENDS else None
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if backend and len(cells) == 10 and re.fullmatch(r"`[a-z0-9_]+`", cells[1]):
            preset = cells[1].strip("`")
            key = backend, preset
            if key not in catalog or key in points:
                raise ValueError(f"Unknown/duplicate ledger row: {key}")
            points[key] = measured_point(
                backend, preset, catalog[key]["spec"]["category"], cells[6], cells[8]
            )
        if cells[0] == "Native DDIM10/H16":
            if h16 is not None or len(cells) != 7:
                raise ValueError("Ambiguous native H16 control")
            h16 = measured_point("cogact", "native_h16", "native_control", cells[4], cells[2], True)
    if points.keys() != catalog.keys():
        raise ValueError(f"Primary coverage mismatch: {catalog.keys() ^ points.keys()}")
    if h16 is None:
        raise ValueError("Missing separately measured native H16 control")
    for backend in BACKENDS:
        if points[backend, "baseline"].speedup != 1.0:
            raise ValueError("Native baseline must be 1×")
    return [points[key] for key in sorted(points)] + [h16]


def render(points, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator, StrMethodFormatter

    labels = {
        "openvla_oft": {
            "baseline": ("Native H8", (-12, 18)),
            "tome": ("ToMe", (-12, 12)),
            "awq": ("AWQ", (10, 20)),
            "safe_pruner": ("SAFE-Pruner", (-6, -22)),
            "fastv": ("FastV", (12, -12)),
        },
        "cogact": {
            "baseline": ("Native H8", (-18, 25)),
            "divprune": ("DivPrune", (-8, 15)),
            "dart": ("DART", (15, 15)),
            "autohorizon": ("AutoHorizon", (7, 27)),
            "pace": ("PACE", (-15, -28)),
            "native_h16": ("Native H16", (-1, 18)),
        },
    }
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.fonttype": "none",
            "svg.hashsalt": "efficient-vla-benchmark-results",
            "text.color": "#243246",
            "axes.labelcolor": "#243246",
            "xtick.color": "#526174",
            "ytick.color": "#526174",
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(left=0.06, right=0.985, top=0.92, bottom=0.20, wspace=0.16)
        for ax, (backend, title) in zip(axes, BACKENDS.items(), strict=True):
            subset = [point for point in points if point.backend == backend]
            baseline = next(point for point in subset if point.preset == "baseline")
            ax.set_title(title, loc="left", size=13, weight="bold", pad=8)
            ax.set_facecolor("#FBFCFE")
            ax.set_axisbelow(True)
            ax.grid(color="#E6EAF0", linewidth=0.7)
            ax.axvline(1, color="#8C97A6", linestyle=(0, (4, 4)), linewidth=1)
            ax.axhline(baseline.success, color="#8C97A6", linestyle=(0, (2, 4)), linewidth=0.8)
            for point in subset:
                if point.category == "baseline":
                    color, marker, size = "#243246", "*", 135
                elif point.supplementary:
                    color, marker, size = "#7B8796", "D", 95
                else:
                    _, color, marker = STYLES[point.category]
                    size = 45
                dot = ax.scatter(
                    point.speedup,
                    point.success,
                    c="none" if point.supplementary else color,
                    marker=marker,
                    s=size,
                    edgecolors=color if point.supplementary else "white",
                    linewidths=1.3 if point.supplementary else 0.65,
                    zorder=5 if point.category == "baseline" else 3,
                )
                kind = "control" if point.supplementary else "primary"
                dot.set_gid(f"{kind}-{backend}-{point.preset}")
                if point.preset in labels[backend]:
                    label, offset = labels[backend][point.preset]
                    ax.annotate(
                        label,
                        (point.speedup, point.success),
                        xytext=offset,
                        textcoords="offset points",
                        ha="right" if offset[0] < 0 else "left",
                        va="center",
                        fontsize=9.5,
                        color="#243246",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
                        arrowprops={"arrowstyle": "-", "color": "#9AA2AF", "lw": 0.7},
                        zorder=7,
                    )
            left, right = (0.47, 1.62) if backend == "openvla_oft" else (0.43, 2.07)
            ax.set_xlim(
                min(left, min(point.speedup for point in subset) - 0.05),
                max(right, max(point.speedup for point in subset) + 0.06),
            )
            lower, upper = (92.9, 97.65) if backend == "openvla_oft" else (84.0, 92.0)
            ax.set_ylim(
                min(lower, min(point.success for point in subset) - 0.3),
                max(upper, max(point.success for point in subset) + 0.4),
            )
            ax.set_xlabel("Speedup (×)", labelpad=5)
            if backend == "openvla_oft":
                ax.set_ylabel("Success rate (%)", labelpad=5)
            ax.xaxis.set_major_locator(MultipleLocator(0.25))
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
            ax.yaxis.set_major_locator(MultipleLocator(1 if backend == "openvla_oft" else 2))
            ax.tick_params(labelsize=10)
            for name in ("top", "right"):
                ax.spines[name].set_visible(False)
            for name in ("left", "bottom"):
                ax.spines[name].set_color("#CCD3DE")
        handles = [
            Line2D([], [], marker=marker, color=color, linestyle="", markersize=7, label=label)
            for label, color, marker in STYLES.values()
        ]
        handles.extend(
            [
                Line2D(
                    [],
                    [],
                    marker="*",
                    color="#243246",
                    linestyle="",
                    markersize=11,
                    label="Native H8 baseline",
                ),
                Line2D(
                    [],
                    [],
                    marker="D",
                    color="#7B8796",
                    markerfacecolor="none",
                    linestyle="",
                    markersize=7,
                    label="Native H16 control",
                ),
            ]
        )
        # Matplotlib fills columns; reorder so readers scan categories left to right.
        handles = [
            handles[index] for column in range(5) for index in range(column, len(handles), 5)
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.52, 0.005),
            ncol=5,
            frameon=False,
            fontsize=9.5,
            columnspacing=1.2,
            handletextpad=0.4,
            labelspacing=0.5,
            borderaxespad=0,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"Creator": "efficient-vla-benchmark"}
        if output.suffix.lower() == ".svg":
            metadata.update(
                Date=None,
                Title="Efficient VLA Benchmark: success and full-cost speedup",
                Description=(
                    "Every primary configuration from docs/benchmark_results.md, plus the "
                    "separate CogACT native H16 control. Same-GPU paired speedups are read "
                    "directly, not recomputed from the displayed native latency."
                ),
            )
            buffer = io.StringIO()
            fig.savefig(buffer, format="svg", metadata=metadata)
            output.write_text(
                "\n".join(line.rstrip() for line in buffer.getvalue().splitlines()) + "\n"
            )
        else:
            fig.savefig(output, metadata=metadata, dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("assets/benchmark_results.svg"))
    args = parser.parse_args()
    if args.output.suffix.lower() not in {".svg", ".png"}:
        parser.error("--output must be SVG (publication) or PNG (local preview)")
    root = Path(__file__).resolve().parents[1]
    points = load_points(root)
    render(points, args.output)
    print(f"Rendered {len(points) - 1} primary configurations and one supplementary control")


if __name__ == "__main__":
    main()
