# qhedge-risk

**A quantum-style (quantum-inspired / quantum-ready) risk / VaR / scenario engine for hedge-style portfolios.**

`qhedge-risk` is an open-source Rust crate for math-heavy, correctness-first risk infrastructure aimed at elite hedge funds, systematic risk teams, and quantitative researchers who want a Rust-native core for:

- Value-at-Risk (VaR)
- Conditional Value-at-Risk (CVaR)
- scenario simulation
- tail-risk and extreme-event stress analysis
- historical replay / stress bundles
- factor-model-based scenario generation
- multiple VaR methodologies: Monte Carlo, historical, and parametric
- correlated multivariate simulation and fat-tail shock models
- future quantum-style / quantum-ready backend experimentation

## What “Q” means

In `qhedge-risk`, **Q** means **quantum-style / quantum-inspired / quantum-ready**, not merely “quant”.

That means:
- the engine runs fully on classical hardware today
- the API supports classical Monte Carlo, quantum-inspired variance reduction, and optional future hardware backends
- ideas such as amplitude-estimation-style sampling, importance sampling, low-discrepancy exploration, and tensor-network-style approximations can be layered into the same risk framework

Quantum hardware is **optional**. Most users will run classical or quantum-inspired backends on CPU/GPU infrastructure.

## Who should use this

- quant developers at top hedge funds
- systematic risk and scenario teams
- portfolio/risk researchers who care about correctness, reproducibility, and performance
- teams interested in future-ready risk infrastructure with a path toward higher assurance via Lean 4

## Core Rust API

Target API shape:

```rust
use chrono::Duration;
use qhedge_risk::{Position, QBackend, RiskEngine, VaRPortfolio};

let portfolio = VaRPortfolio::from_positions(
    vec![
        Position {
            instrument: "ES_FUT".into(),
            quantity: 10.0,
            price: 5000.0,
            volatility: 0.22,
            delta: 1.0,
        },
    ],
    vec![1.0],
);

let engine = RiskEngine::new(
    QBackend::QuantumInspired { max_samples: 8_192 },
    0.99,
)?;

let var = engine.compute_var(&portfolio, Duration::days(1))?;
let cvar = engine.compute_cvar(&portfolio, Duration::days(1))?;
let bundle = engine.simulate_scenarios(&portfolio, 10_000)?;
```

## Backend model

The crate exposes a backend abstraction:

- `QBackend::ClassicalFallback`
- `QBackend::QuantumInspired { max_samples }`
- `QBackend::QuantumHardware { leap_token, solver_url }`

The current implementation focuses on classical and quantum-inspired runtime paths, with shock models such as Gaussian, Student-t, and crash-mixture style tails. The hardware path is a future-ready interface placeholder.

## Lean verification layer

A `lean/` directory is included to support a high-assurance future:

- `Portfolio` and `ReturnProcess` as Lean types
- `VaR` and `CVaR` as pure functions
- toy lemmas such as:
  - `cvar_geq_var`
  - `monotonic_in_exposure`

This Lean layer is optional for users, but valuable for institutional-grade correctness narratives.

## Python integration sketch

A `python/` layout sketch is included for PyO3/maturin bindings so quants can call risk methods from notebooks and research environments.

## Repository layout

```text
qhedge-risk/
├── Cargo.toml
├── README.md
├── src/
│   └── lib.rs
├── lean/
│   └── QhedgeRisk.lean
└── python/
    └── README.md
```

## Status

This is an initial scaffold for a future open-source quant risk engine with:
- mathematically clean Rust APIs
- quantum-inspired backend abstractions
- Lean-verification direction
- Python integration path

## Mathematical focus

The engine now supports:
- correlated covariance-driven Monte Carlo
- factor-model shock propagation
- historical replay
- parametric Gaussian VaR
- heavy-tail stress generation via Student-t and crash-mixture shock models
