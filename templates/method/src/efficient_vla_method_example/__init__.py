"""Metadata only: implementation imports remain lazy."""


def register():
    return {
        "schema_version": 1,
        "name": "example_scale",
        "category": "example",
        "config": "efficient_vla_method_example.config:Config",
        "description": "Contract fixture, not an acceleration algorithm.",
        "implementations": {
            "example": {
                "variants": ["long_chunk"],
                "requires": ["action_scale"],
                "factory": "efficient_vla_method_example.method:build",
            }
        },
    }
