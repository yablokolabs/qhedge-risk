# Python binding sketch for qhedge-risk

Recommended structure:

```text
python/
├── Cargo.toml          # PyO3 crate manifest
├── pyproject.toml      # maturin configuration
└── src/
    └── lib.rs          # Python wrappers around qhedge_risk Rust APIs
```

Suggested `Cargo.toml` shape:

```toml
[package]
name = "qhedge-risk-python"
version = "0.1.0"
edition = "2024"

[lib]
name = "qhedge_risk"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
serde_json = "1"
qhedge-risk = { path = "../", features = ["python_binding"] }
```

Suggested `pyproject.toml` shape:

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "qhedge-risk"
version = "0.1.0"
description = "Quantum-style risk / VaR / scenario engine for hedge-style portfolios"
requires-python = ">=3.10"
```

Suggested wrapper sketch:

```rust
use pyo3::prelude::*;

#[pyfunction]
fn compute_risk() -> PyResult<String> {
    Ok("wire RiskEngine here".to_string())
}

#[pymodule]
fn qhedge_risk(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_risk, m)?)?;
    Ok(())
}
```

Recommended initial Python-facing API:
- `compute_var(...)`
- `compute_cvar(...)`
- `simulate_scenarios(...)`
- `compute_tail_risk(...)`
