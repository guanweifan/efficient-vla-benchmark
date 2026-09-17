"""Discover optional plugins and paper metadata without importing model runtimes."""

import importlib
import importlib.metadata
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import yaml

from efficient_vla_benchmark.core.backend import API_VERSION, BackendSpec, RuntimeResources
from efficient_vla_benchmark.core.config import IDENTIFIER, ExperimentConfig

BACKEND_GROUP = "efficient_vla_benchmark.backends"
METHOD_GROUP = "efficient_vla_benchmark.methods"


def resolve(reference: str):
    module, separator, attribute = reference.partition(":")
    if not separator or not module or not attribute:
        raise ValueError("An implementation entry must have module:attribute form")
    return getattr(importlib.import_module(module), attribute)


def backend_entries() -> dict:
    entries = {}
    for entry in importlib.metadata.entry_points(group=BACKEND_GROUP):
        if entry.name in entries:
            raise ValueError(f"Duplicate backend plugin: {entry.name}")
        entries[entry.name] = entry
    return entries


def backend_spec(name: str) -> BackendSpec:
    entry = backend_entries().get(name)
    if entry is None:
        raise ValueError(f"Backend {name!r} is not installed; install its optional plugin")
    spec = entry.load()()
    if not isinstance(spec, BackendSpec) or spec.name != name or spec.api_version != API_VERSION:
        raise ValueError("Backend plugin has an incompatible interface version or identity")
    return spec


@dataclass(frozen=True)
class MethodSpec:
    name: str
    category: str
    config: str
    implementations: dict
    description: str = ""
    requires_artifact: bool = False


def parse_method(data: dict) -> MethodSpec:
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Unsupported method metadata schema")
    for key in ("name", "category"):
        if not isinstance(data.get(key), str) or not IDENTIFIER.fullmatch(data[key]):
            raise ValueError("Method names and categories must use snake_case")
    if not isinstance(data.get("config"), str) or ":" not in data["config"]:
        raise ValueError("Method metadata must declare a configuration entry")
    if type(data.get("requires_artifact", False)) is not bool:
        raise ValueError("requires_artifact must be boolean")
    implementations = data.get("implementations")
    if not isinstance(implementations, dict) or not implementations:
        raise ValueError("A method must have at least one explicit backend implementation")
    for backend, implementation in implementations.items():
        if (
            not isinstance(backend, str)
            or not IDENTIFIER.fullmatch(backend)
            or not isinstance(implementation, dict)
        ):
            raise ValueError("Invalid backend implementation metadata")
        variants = implementation.get("variants")
        if (
            not isinstance(variants, list)
            or not variants
            or any(not isinstance(item, str) or not IDENTIFIER.fullmatch(item) for item in variants)
        ):
            raise ValueError("Implementations declare an explicit list of model variants")
        if len(set(variants)) != len(variants):
            raise ValueError("Model variants must be unique")
        factory = implementation.get("factory")
        if not isinstance(factory, str) or ":" not in factory:
            raise ValueError("Implementations declare model variants and a factory")
        required = implementation.get("requires", [])
        if not isinstance(required, list) or any(
            not isinstance(item, str) or not IDENTIFIER.fullmatch(item) for item in required
        ):
            raise ValueError("Required capabilities must be an explicit list")
    return MethodSpec(
        data["name"],
        data["category"],
        data["config"],
        implementations,
        data.get("description", ""),
        bool(data.get("requires_artifact", False)),
    )


def method_catalog() -> dict[str, MethodSpec]:
    catalog = {}
    root = files("efficient_vla_benchmark").joinpath("methods")
    if root.is_dir():
        for category in root.iterdir():
            if not category.is_dir() or category.name.startswith("_"):
                continue
            for method in category.iterdir():
                metadata = method.joinpath("metadata.yaml")
                if method.is_dir() and metadata.is_file():
                    spec = parse_method(yaml.safe_load(metadata.read_text()))
                    if spec.category != category.name or spec.name != method.name:
                        raise ValueError("Method directory and metadata identities disagree")
                    if spec.name in catalog:
                        raise ValueError("Duplicate method identity")
                    catalog[spec.name] = spec
    for entry in importlib.metadata.entry_points(group=METHOD_GROUP):
        if entry.name in catalog:
            raise ValueError("Duplicate external method identity")
        spec = parse_method(entry.load()())
        if spec.name != entry.name:
            raise ValueError("External method identity disagrees with its entry point")
        catalog[spec.name] = spec
    return catalog


def select_method(config: ExperimentConfig, spec: BackendSpec):
    if config.method == "none":
        if config.parameters:
            raise ValueError("Native baseline does not accept method parameters")
        return None, None, None
    method = method_catalog().get(config.method)
    if method is None:
        raise ValueError(f"Method {config.method!r} has not been integrated")
    implementation = method.implementations.get(config.backend)
    if not implementation or config.model_variant not in implementation["variants"]:
        raise ValueError("Unsupported method/backend/model-variant combination")
    required = set(implementation.get("requires", []))
    if not required.issubset(spec.capabilities):
        raise ValueError("Backend lacks capabilities required by this method")
    method_config = resolve(method.config).parse(config.parameters)
    if implementation.get("validate_experiment"):
        resolve(implementation["validate_experiment"])(config, method_config)
    return method, implementation, method_config


def validate_selection(config: ExperimentConfig):
    spec = backend_spec(config.backend)
    if config.model_variant not in spec.variants:
        raise ValueError("Unsupported model variant for the selected backend")
    resolve(spec.validator)(config)
    return spec, *select_method(config, spec)


def build_backend(
    config: ExperimentConfig, model_path: Path, device: str, artifact_path: Path | None = None
):
    spec, method, implementation, method_config = validate_selection(config)
    resources = RuntimeResources(model_path, device, artifact_path)
    if artifact_path is not None and not (method and method.requires_artifact):
        raise ValueError("This method does not consume an artifact")
    if method and method.requires_artifact and artifact_path is None:
        raise ValueError("The selected method requires --artifact-path")
    if implementation and implementation.get("preflight"):
        resolve(implementation["preflight"])(method_config, resources)
    backend = resolve(spec.factory)(config, resources)
    try:
        if implementation:
            backend.attach_method(
                resolve(implementation["factory"])(backend, method_config, resources)
            )
        return backend
    except Exception:
        backend.close()
        raise


def calibrate_backend(config: ExperimentConfig, resources: RuntimeResources):
    spec, method, implementation, method_config = validate_selection(config)
    if not method or not implementation.get("calibrate"):
        raise ValueError("The selected implementation does not expose offline calibration")
    if resources.dataset_path is None or resources.artifact_dir is None:
        raise ValueError("Calibration needs explicit dataset and artifact destinations")
    backend = resolve(spec.factory)(config, resources)
    try:
        return resolve(implementation["calibrate"])(backend, method_config, resources)
    finally:
        backend.close()
