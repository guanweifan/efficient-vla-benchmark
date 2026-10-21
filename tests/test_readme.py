"""Keep the public entry point aligned with measured presets and scope metadata."""

import importlib.util
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "plot_benchmark_results", ROOT / "scripts/plot_benchmark_results.py"
)
plot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = plot
SPEC.loader.exec_module(plot)


def test_method_matrix_matches_every_non_native_primary_once():
    catalog = plot.selection_catalog(ROOT)
    expected = {value["path"]: value for value in catalog.values() if value["readme"]}
    readme = (ROOT / "README.md").read_text()
    found = []
    groups = {style[0]: category for category, style in plot.STYLES.items()}
    category = None
    for line in readme.splitlines():
        summary = re.fullmatch(r"<summary><strong>(.+)</strong></summary>", line)
        if summary:
            category = groups[summary[1]]
        if not line.startswith("| ["):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        assert len(cells) == 4
        owner = re.fullmatch(r"\[.+\]\((.+/README.md)\)", cells[0])
        assert owner
        for backend, cell in zip(plot.BACKENDS, cells[1:3], strict=True):
            if cell == "—":
                continue
            link = re.fullmatch(r"\[✓\]\((.+\.yaml)\)", cell)
            assert link
            path = Path(link[1])
            assert path.parent.name == backend
            entry = expected[path]
            assert entry["spec"]["category"] == category
            assert ROOT / owner[1] == entry["readme"]
            assert not entry["spec"].get("requires_artifact") or cells[3] != "None"
            found.append(path)
    assert Counter(found) == Counter({path: 1 for path in expected})


def test_figure_uses_all_final_results_and_paired_speedups():
    points = plot.load_points(ROOT)
    primary = [point for point in points if not point.supplementary]
    assert {point.key for point in primary} == set(plot.selection_catalog(ROOT))
    assert len([point for point in primary if point.category == "baseline"]) == len(plot.BACKENDS)
    by_key = {point.key: point for point in points}
    # Read the paired speedup, not 23.399 / 17.214 from the displayed CogACT native row.
    assert by_key["cogact", "awq"].speedup == 1.369
    assert by_key["cogact", "awq"].successes == 1804
    assert by_key["openvla_oft", "baseline"].successes == 1932
    h16 = by_key["cogact", "native_h16"]
    assert h16.supplementary and h16.successes == 1799 and h16.speedup == 1.961


@pytest.mark.parametrize(
    "speed,success",
    [
        ("—", "90.60% (1812)"),
        ("0.000×", "90.60% (1812)"),
        ("1.000×", "90.60% (1813)"),
        ("1.000×", "100.05% (2001)"),
        ("nan×", "90.60% (1812)"),
    ],
)
def test_figure_rejects_missing_or_inconsistent_measurements(speed, success):
    with pytest.raises(ValueError):
        plot.measured_point("cogact", "test", "caching", speed, success)


def test_published_svg_contains_every_point_and_no_embedded_image():
    svg = ET.parse(ROOT / "assets/benchmark_results.svg")
    nodes = list(svg.iter())
    ids = [node.get("id", "") for node in nodes]
    expected = {
        f"{'control' if point.supplementary else 'primary'}-{point.backend}-{point.preset}"
        for point in plot.load_points(ROOT)
    }
    assert {key for key in ids if key.startswith(("primary-", "control-"))} == expected
    assert len(ids) - ids.count("") == len(set(ids) - {""})
    assert not any(node.tag.endswith(("}image", "}script", "}foreignObject")) for node in nodes)
    assert not any(
        "file:" in value or "data:" in value for node in nodes for value in node.attrib.values()
    )


def test_readme_local_navigation():
    text = (ROOT / "README.md").read_text()
    headings = re.findall(r"^#{1,6} (.+)$", text, re.M)
    anchors = {re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-") for heading in headings}
    assert set(re.findall(r'href="#([^"]+)"', text)) <= anchors
    assert text.count("<details>") == text.count("</details>")
    assert text.count("<summary>") == text.count("</summary>")
