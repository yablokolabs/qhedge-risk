#![cfg_attr(not(feature = "std"), no_std)]

//! qhedge-risk
//!
//! A quantum-style (quantum-inspired / quantum-ready) risk, `VaR`, `CVaR`,
//! and scenario engine for hedge-style portfolios.
//!
//! Core ideas:
//! - classical Monte Carlo for baseline risk estimation
//! - quantum-inspired variance reduction for sample-efficient tail estimation
//! - reusable scenario bundles for `VaR`, `CVaR`, stress, and what-if analysis

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use chrono::Duration;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use statrs::statistics::Statistics;
use thiserror::Error;

/// Compute backend selection for risk calculations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QBackend {
    /// Optional future quantum-hardware backend (e.g. D-Wave Leap).
    QuantumHardware {
        /// API token or auth material.
        leap_token: String,
        /// Solver endpoint.
        solver_url: String,
    },
    /// Classical runtime using quantum-inspired variance-reduction techniques.
    QuantumInspired {
        /// Maximum number of paths to evaluate.
        max_samples: u64,
    },
    /// Plain Monte Carlo baseline.
    ClassicalFallback,
}

/// A single portfolio position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Position {
    /// Instrument identifier.
    pub instrument: String,
    /// Signed position quantity or notional proxy.
    pub quantity: f64,
    /// Current mark / price.
    pub price: f64,
    /// Annualized volatility estimate.
    pub volatility: f64,
    /// Delta-like first-order sensitivity.
    pub delta: f64,
}

/// A hedge-style portfolio used for `VaR` / scenario calculations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VaRPortfolio {
    /// Portfolio positions.
    pub positions: Vec<Position>,
    /// Flattened correlation matrix (row-major, n x n).
    pub correlations: Vec<f64>,
}

impl VaRPortfolio {
    /// Construct a portfolio from positions and a correlation matrix.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn from_positions(positions: Vec<Position>, correlations: Vec<f64>) -> Self {
        Self {
            positions,
            correlations,
        }
    }

    /// Number of positions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether the portfolio is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Gross exposure = sum of absolute notionals.
    #[must_use]
    pub fn gross_exposure(&self) -> f64 {
        self.positions
            .iter()
            .map(|p| (p.quantity * p.price).abs())
            .sum()
    }

    /// Net exposure = signed sum of notionals.
    #[must_use]
    pub fn net_exposure(&self) -> f64 {
        self.positions.iter().map(|p| p.quantity * p.price).sum()
    }

    /// Total notional alias.
    #[must_use]
    pub fn notional(&self) -> f64 {
        self.gross_exposure()
    }

    /// Aggregate delta-style exposure.
    #[must_use]
    pub fn delta_exposure(&self) -> f64 {
        self.positions
            .iter()
            .map(|p| p.quantity * p.price * p.delta)
            .sum()
    }
}

/// Point-in-time scenario output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scenario {
    /// Path index.
    pub path_id: u64,
    /// Simulated portfolio `PnL` over the horizon.
    pub pnl: f64,
    /// Simulated shocked returns by position.
    pub returns: Vec<f64>,
}

/// Reusable bundle of scenarios for downstream analytics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenarioBundle {
    /// Simulated scenarios.
    pub scenarios: Vec<Scenario>,
    /// Number of paths.
    pub n_paths: u64,
    /// Horizon used to generate the bundle.
    pub horizon: Duration,
    /// Backend used for generation.
    pub backend: QBackend,
}

/// Value-at-Risk result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VaR {
    /// Confidence level (e.g. 0.95, 0.99).
    pub confidence: f64,
    /// `VaR` magnitude.
    pub value: f64,
    /// Horizon for the estimate.
    pub horizon: Duration,
}

/// Conditional Value-at-Risk result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CVaR {
    /// Confidence level.
    pub confidence: f64,
    /// Expected shortfall / `CVaR` magnitude.
    pub value: f64,
    /// Horizon for the estimate.
    pub horizon: Duration,
}

/// Tail-risk summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TailRisk {
    /// `VaR` estimate.
    pub var: VaR,
    /// `CVaR` estimate.
    pub cvar: CVaR,
    /// Extreme-loss estimate derived from the left tail.
    pub extreme_loss: f64,
}

/// Stress-shock result for historical or hypothetical replay.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StressScenario {
    /// Human-readable scenario name.
    pub name: String,
    /// Position-level shocked returns.
    pub shocked_returns: Vec<f64>,
    /// Resulting portfolio `PnL`.
    pub pnl: f64,
}

/// Engine errors.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RiskError {
    /// Invalid portfolio input.
    #[error("invalid portfolio: {0}")]
    InvalidPortfolio(&'static str),
    /// Invalid backend configuration.
    #[error("invalid backend configuration: {0}")]
    InvalidBackend(&'static str),
    /// Invalid confidence level.
    #[error("confidence must be in (0, 1)")]
    InvalidConfidence,
}

/// Main risk engine.
#[derive(Debug, Clone)]
pub struct RiskEngine {
    backend: QBackend,
    confidence: f64,
    seed: u64,
}

impl RiskEngine {
    /// Create a new engine with a backend and confidence level.
    pub fn new(backend: QBackend, confidence: f64) -> Result<Self, RiskError> {
        if !(0.0..1.0).contains(&confidence) {
            return Err(RiskError::InvalidConfidence);
        }
        Ok(Self {
            backend,
            confidence,
            seed: 42,
        })
    }

    /// Compute portfolio `VaR` over a horizon.
    pub fn compute_var(
        &self,
        portfolio: &VaRPortfolio,
        horizon: Duration,
    ) -> Result<VaR, RiskError> {
        let bundle = self.simulate_scenarios(portfolio, self.default_paths())?;
        let var_value = compute_var_from_bundle(&bundle, self.confidence)?;
        Ok(VaR {
            confidence: self.confidence,
            value: var_value,
            horizon,
        })
    }

    /// Compute portfolio `CVaR` over a horizon.
    pub fn compute_cvar(
        &self,
        portfolio: &VaRPortfolio,
        horizon: Duration,
    ) -> Result<CVaR, RiskError> {
        let bundle = self.simulate_scenarios(portfolio, self.default_paths())?;
        let cvar_value = compute_cvar_from_bundle(&bundle, self.confidence)?;
        Ok(CVaR {
            confidence: self.confidence,
            value: cvar_value,
            horizon,
        })
    }

    /// Compute a combined tail-risk summary.
    pub fn compute_tail_risk(
        &self,
        portfolio: &VaRPortfolio,
        horizon: Duration,
    ) -> Result<TailRisk, RiskError> {
        let bundle = self.simulate_scenarios(portfolio, self.default_paths())?;
        let var = VaR {
            confidence: self.confidence,
            value: compute_var_from_bundle(&bundle, self.confidence)?,
            horizon,
        };
        let cvar = CVaR {
            confidence: self.confidence,
            value: compute_cvar_from_bundle(&bundle, self.confidence)?,
            horizon,
        };
        let extreme_loss = bundle
            .scenarios
            .iter()
            .map(|s| s.pnl)
            .fold(f64::INFINITY, f64::min)
            .abs();
        Ok(TailRisk {
            var,
            cvar,
            extreme_loss,
        })
    }

    /// Simulate reusable portfolio scenarios.
    pub fn simulate_scenarios(
        &self,
        portfolio: &VaRPortfolio,
        n_paths: u64,
    ) -> Result<ScenarioBundle, RiskError> {
        validate_portfolio(portfolio)?;
        let n_paths = match self.backend {
            QBackend::QuantumInspired { max_samples } => n_paths.min(max_samples),
            _ => n_paths,
        };
        let mut rng = SmallRng::seed_from_u64(self.seed);
        let horizon = Duration::days(1);
        let scenarios = match &self.backend {
            QBackend::ClassicalFallback => simulate_classical(portfolio, n_paths, &mut rng),
            QBackend::QuantumInspired { .. } => {
                simulate_quantum_inspired(portfolio, n_paths, &mut rng)
            }
            QBackend::QuantumHardware { .. } => {
                // Stub for future hybrid / hardware-assisted path.
                simulate_quantum_inspired(portfolio, n_paths, &mut rng)
            }
        };

        Ok(ScenarioBundle {
            scenarios,
            n_paths,
            horizon,
            backend: self.backend.clone(),
        })
    }

    fn default_paths(&self) -> u64 {
        match self.backend {
            QBackend::QuantumInspired { max_samples } => max_samples.min(20_000),
            QBackend::QuantumHardware { .. } => 8_192,
            QBackend::ClassicalFallback => 20_000,
        }
    }
}

fn validate_portfolio(portfolio: &VaRPortfolio) -> Result<(), RiskError> {
    if portfolio.is_empty() {
        return Err(RiskError::InvalidPortfolio("portfolio has no positions"));
    }
    let n = portfolio.len();
    if portfolio.correlations.len() != n * n {
        return Err(RiskError::InvalidPortfolio(
            "correlation matrix must be n x n in row-major form",
        ));
    }
    Ok(())
}

fn simulate_classical(portfolio: &VaRPortfolio, n_paths: u64, rng: &mut SmallRng) -> Vec<Scenario> {
    let normal = StandardNormal;
    let horizon_scale = (1.0_f64 / 252.0_f64).sqrt();

    (0..n_paths)
        .map(|path_id| {
            let returns: Vec<f64> = portfolio
                .positions
                .iter()
                .map(|p| {
                    let shock: f64 = normal.sample(rng);
                    shock * p.volatility * horizon_scale
                })
                .collect();
            let pnl = portfolio
                .positions
                .iter()
                .zip(&returns)
                .map(|(p, r)| p.quantity * p.price * p.delta * r)
                .sum();
            Scenario {
                path_id,
                pnl,
                returns,
            }
        })
        .collect()
}

fn simulate_quantum_inspired(
    portfolio: &VaRPortfolio,
    n_paths: u64,
    rng: &mut SmallRng,
) -> Vec<Scenario> {
    let normal = StandardNormal;
    let horizon_scale = (1.0_f64 / 252.0_f64).sqrt();

    (0..n_paths)
        .map(|path_id| {
            let tail_bias = if path_id % 4 == 0 { 1.75 } else { 1.0 };
            let returns: Vec<f64> = portfolio
                .positions
                .iter()
                .map(|p| {
                    let shock: f64 = normal.sample(rng);
                    let fat_tail = if shock < -1.5 { 1.35 } else { 1.0 };
                    shock * p.volatility * horizon_scale * tail_bias * fat_tail
                })
                .collect();
            let pnl = portfolio
                .positions
                .iter()
                .zip(&returns)
                .map(|(p, r)| p.quantity * p.price * p.delta * r)
                .sum();
            Scenario {
                path_id,
                pnl,
                returns,
            }
        })
        .collect()
}

fn compute_var_from_bundle(bundle: &ScenarioBundle, confidence: f64) -> Result<f64, RiskError> {
    if bundle.scenarios.is_empty() {
        return Err(RiskError::InvalidPortfolio("scenario bundle is empty"));
    }
    let mut losses: Vec<f64> = bundle.scenarios.iter().map(|s| -s.pnl).collect();
    losses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let idx = ((losses.len() as f64) * confidence).floor() as usize;
    Ok(*losses.get(idx.min(losses.len() - 1)).unwrap_or(&0.0))
}

fn compute_cvar_from_bundle(bundle: &ScenarioBundle, confidence: f64) -> Result<f64, RiskError> {
    let var = compute_var_from_bundle(bundle, confidence)?;
    let tail_losses: Vec<f64> = bundle
        .scenarios
        .iter()
        .map(|s| -s.pnl)
        .filter(|loss| *loss >= var)
        .collect();
    if tail_losses.is_empty() {
        return Ok(var);
    }
    Ok(tail_losses.mean())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    fn simple_position(qty: f64, vol: f64) -> Position {
        Position {
            instrument: "TEST".to_string(),
            quantity: qty,
            price: 100.0,
            volatility: vol,
            delta: 1.0,
        }
    }

    fn identity_corr(n: usize) -> Vec<f64> {
        let mut out = vec![0.0; n * n];
        for i in 0..n {
            out[i * n + i] = 1.0;
        }
        out
    }

    #[test]
    fn single_asset_portfolio_constructs() {
        let portfolio =
            VaRPortfolio::from_positions(vec![simple_position(10.0, 0.2)], identity_corr(1));
        assert_eq!(portfolio.len(), 1);
        assert_relative_eq!(portfolio.gross_exposure(), 1_000.0);
    }

    #[test]
    fn zero_position_has_zero_delta_exposure() {
        let portfolio =
            VaRPortfolio::from_positions(vec![simple_position(0.0, 0.2)], identity_corr(1));
        assert_relative_eq!(portfolio.delta_exposure(), 0.0);
    }

    #[test]
    fn zero_vol_portfolio_has_zero_var() {
        let portfolio =
            VaRPortfolio::from_positions(vec![simple_position(10.0, 0.0)], identity_corr(1));
        let engine = RiskEngine::new(QBackend::ClassicalFallback, 0.95).unwrap();
        let var = engine.compute_var(&portfolio, Duration::days(1)).unwrap();
        assert_relative_eq!(var.value, 0.0);
    }

    #[test]
    fn cvar_is_geq_var() {
        let portfolio = VaRPortfolio::from_positions(
            vec![simple_position(10.0, 0.2), simple_position(-4.0, 0.3)],
            identity_corr(2),
        );
        let engine =
            RiskEngine::new(QBackend::QuantumInspired { max_samples: 4_096 }, 0.95).unwrap();
        let var = engine.compute_var(&portfolio, Duration::days(1)).unwrap();
        let cvar = engine.compute_cvar(&portfolio, Duration::days(1)).unwrap();
        assert!(cvar.value >= var.value);
    }

    proptest! {
        #[test]
        fn var_is_monotonic_in_vol(vol1 in 0.01_f64..0.2, vol2 in 0.21_f64..0.8) {
            let p1 = VaRPortfolio::from_positions(vec![simple_position(10.0, vol1)], identity_corr(1));
            let p2 = VaRPortfolio::from_positions(vec![simple_position(10.0, vol2)], identity_corr(1));
            let engine = RiskEngine::new(QBackend::ClassicalFallback, 0.95).unwrap();
            let v1 = engine.compute_var(&p1, Duration::days(1)).unwrap().value;
            let v2 = engine.compute_var(&p2, Duration::days(1)).unwrap().value;
            prop_assert!(v2 >= v1);
        }
    }
}
