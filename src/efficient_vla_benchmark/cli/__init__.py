"""Lightweight command entrypoint; model runtimes are always loaded on demand."""

import argparse


def main():
    parser = argparse.ArgumentParser(prog="evb")
    parser.add_argument("--version", action="version", version="efficient-vla-benchmark 0.2.0.dev0")
    parser.parse_args()

