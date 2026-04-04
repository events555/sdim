# Run all checks (lint, typecheck, test+coverage)
check: lint typecheck test

lint:
    uv run ruff check .

typecheck:
    uv run ty check sdim/ tests/

test:
    uv run pytest tests/ -x --ignore=tests/test_circuit.py --ignore=tests/test_gate.py --ignore=tests/test_tomography.py

# Run tests including cirq-dependent ones (requires sdim[interop])
test-all:
    uv run pytest tests/ -x

# Run benchmarks (saves JSON baseline for Rust comparison)
bench:
    uv run pytest benchmarks/ --benchmark-only --benchmark-save=baseline -q

# Compile typst documents to PDF in docs/
pdf:
    for f in docs/typst/*.typ; do typst compile "$f" "docs/pdf/$(basename "${f%.typ}.pdf")"; done
