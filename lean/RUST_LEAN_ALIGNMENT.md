# Rust ↔ Lean Alignment Notes

This document explains the intended correspondence between the executable Rust engine and the Lean specification sketch.

## Scope
This is **not** a proof that the Rust code is correct.
It is a concept/spec alignment layer.

## Main mappings

### Shock model
Rust:
- `ShockModel::Gaussian`
- `ShockModel::StudentT { degrees_of_freedom }`
- `ShockModel::CrashMixture { crash_probability, shock_multiplier }`

Lean:
- `ShockModel.gaussian`
- `ShockModel.studentT`
- `ShockModel.crashMixture`

### VaR method
Rust:
- `VaRMethod::MonteCarlo`
- `VaRMethod::Historical`
- `VaRMethod::Parametric`

Lean:
- `VaRMethod.monteCarlo`
- `VaRMethod.historical`
- `VaRMethod.parametric`

### Position / portfolio state
Rust `Position` carries:
- first-order exposure via `quantity * price * delta`
- second-order sensitivity via `gamma`
- factor loadings
- idiosyncratic volatility

Lean `Position` mirrors these semantically as:
- `exposure`
- `delta`
- `gamma`
- `factorLoadings`
- `idioVol`

Lean collapses `quantity * price` into a single `exposure` field.

### Delta-gamma loss approximation
Rust:
- `pnl_from_returns(...)`
- delta term + gamma term

Lean:
- `positionLossFromReturn`
- `portfolioLossFromReturns`

Lean models **loss**, while Rust computes `PnL`; the signs are intentionally flipped when aligning risk measures.

### Historical replay
Rust:
- `RiskEngine::replay_historical`
- returns a `ScenarioBundle`

Lean:
- `replayHistorical`
- `historicalScenarioBundle`

### Empirical VaR / CVaR
Rust:
- `compute_var_from_bundle`
- `compute_cvar_from_bundle`

Lean:
- `empiricalVaR`
- `tailLosses`
- `empiricalCVaR`
- `monteCarloVaR`
- `monteCarloCVaR`
- `historicalVaR`
- `historicalCVaR`

## Important gaps still remaining

### 1. Quantile exactness
Rust uses a concrete quantile index over sorted losses.
Lean still uses a simplified empirical quantile placeholder.

### 2. Correlated shock construction
Rust implements covariance assembly + Cholesky-style correlation handling.
Lean does not formalize that sampler yet.

### 3. Runtime validation/error behavior
Rust includes executable error paths and validation.
Lean does not model all error cases.

### 4. Proof completeness
Lean still contains proof placeholders (`sorry`).

## Honest status
Current state is:
- **good concept alignment**
- **partial mathematical alignment**
- **not full verification**
- **not exact algorithmic equivalence yet**
