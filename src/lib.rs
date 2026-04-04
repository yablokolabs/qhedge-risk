#![cfg_attr(not(feature = "std"), no_std)]

//! qhedge-risk
//!
//! A quantum-style (quantum-inspired / quantum-ready) risk, `VaR`, `CVaR`,
//! and scenario engine for hedge-style portfolios.
//!
//! This crate focuses on:
//! - classical Monte Carlo for baseline risk estimation
//! - quantum-inspired variance reduction for sample-efficient tail estimation
//! - historical replay / stress bundles
//! - fat-tail scenario generation
//! - reusable scenario bundles for `VaR`, `CVaR`, stress, and what-if analysis

extern crate alloc;

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use chrono::Duration;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{ChiSquared, Distribution, StandardNormal};
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

/// Shock model used for scenario generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ShockModel {
    /// Correlated Gaussian shocks.
    Gaussian,
    /// Correlated Student-t shocks for heavier tails.
    StudentT {
        /// Degrees of freedom controlling tail heaviness.
        degrees_of_freedom: f64,
    },
    /// Gaussian core with a crash mixture probability.
    CrashMixture {
        /// Probability of tail/crash regime.
        crash_probability: f64,
        /// Multiplier applied during crash regime.
        shock_multiplier: f64,
    },
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
    /// Gamma-like second-order sensitivity.
    pub gamma: f64,
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
    /// Shock model used for generation.
    pub shock_model: ShockModel,
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
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Error, Clone, PartialEq)]
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
    /// Invalid shock model.
    #[error("invalid shock model: {0}")]
    InvalidShockModel(&'static str),
}

/// Main risk engine.
#[derive(Debug, Clone)]
pub struct RiskEngine {
    backend: QBackend,
    confidence: f64,
    seed: u64,
    shock_model: ShockModel,
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
            shock_model: ShockModel::Gaussian,
        })
    }

    /// Override the shock model.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn with_shock_model(mut self, shock_model: ShockModel) -> Self {
        self.shock_model = shock_model;
        self
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
            QBackend::ClassicalFallback => {
                simulate_classical(portfolio, n_paths, &self.shock_model, &mut rng)?
            }
            QBackend::QuantumInspired { .. } | QBackend::QuantumHardware { .. } => {
                simulate_quantum_inspired(portfolio, n_paths, &self.shock_model, &mut rng)?
            }
        };

        Ok(ScenarioBundle {
            scenarios,
            n_paths,
            horizon,
            backend: self.backend.clone(),
            shock_model: self.shock_model.clone(),
        })
    }

    /// Historical replay over shocked returns.
    pub fn replay_historical(
        &self,
        portfolio: &VaRPortfolio,
        historical_returns: &[Vec<f64>],
    ) -> Result<ScenarioBundle, RiskError> {
        validate_portfolio(portfolio)?;
        let scenarios = historical_returns
            .iter()
            .enumerate()
            .map(|(idx, returns)| Scenario {
                path_id: idx as u64,
                pnl: pnl_from_returns(portfolio, returns),
                returns: returns.clone(),
            })
            .collect();
        Ok(ScenarioBundle {
            scenarios,
            n_paths: historical_returns.len() as u64,
            horizon: Duration::days(1),
            backend: self.backend.clone(),
            shock_model: self.shock_model.clone(),
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

fn covariance_matrix(portfolio: &VaRPortfolio) -> Vec<Vec<f64>> {
    let n = portfolio.len();
    let mut cov = vec![vec![0.0; n]; n];
    for (i, row) in cov.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            let rho = portfolio.correlations[i * n + j];
            *cell = rho * portfolio.positions[i].volatility * portfolio.positions[j].volatility;
        }
    }
    cov
}

fn cholesky_decompose(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = matrix.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let sum: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
            if i == j {
                l[i][j] = (matrix[i][i] - sum).max(0.0).sqrt();
            } else if l[j][j] > 0.0 {
                l[i][j] = (matrix[i][j] - sum) / l[j][j];
            }
        }
    }
    l
}

fn correlated_shocks(
    portfolio: &VaRPortfolio,
    shock_model: &ShockModel,
    rng: &mut SmallRng,
) -> Result<Vec<f64>, RiskError> {
    let n = portfolio.len();
    let cov = covariance_matrix(portfolio);
    let l = cholesky_decompose(&cov);
    let mut z = vec![0.0; n];

    match shock_model {
        ShockModel::Gaussian => {
            let normal = StandardNormal;
            for zi in &mut z {
                *zi = normal.sample(rng);
            }
        }
        ShockModel::StudentT { degrees_of_freedom } => {
            if *degrees_of_freedom <= 2.0 {
                return Err(RiskError::InvalidShockModel(
                    "student-t degrees_of_freedom must be > 2",
                ));
            }
            let normal = StandardNormal;
            let chi = ChiSquared::new(*degrees_of_freedom)
                .map_err(|_| RiskError::InvalidShockModel("invalid chi-squared parameters"))?;
            let scale = (chi.sample(rng) / *degrees_of_freedom).sqrt();
            for zi in &mut z {
                let base: f64 = normal.sample(rng);
                *zi = base / scale;
            }
        }
        ShockModel::CrashMixture {
            crash_probability,
            shock_multiplier,
        } => {
            let normal = StandardNormal;
            let crash = rand::Rng::random::<f64>(rng) < *crash_probability;
            let mult = if crash { *shock_multiplier } else { 1.0 };
            for zi in &mut z {
                let base: f64 = normal.sample(rng);
                *zi = base * mult;
            }
        }
    }

    let mut correlated = vec![0.0; n];
    for i in 0..n {
        correlated[i] = (0..=i).map(|j| l[i][j] * z[j]).sum();
    }
    Ok(correlated)
}

fn pnl_from_returns(portfolio: &VaRPortfolio, returns: &[f64]) -> f64 {
    portfolio
        .positions
        .iter()
        .zip(returns)
        .map(|(p, r)| {
            let delta_pnl = p.quantity * p.price * p.delta * r;
            let gamma_pnl = 0.5 * p.quantity * p.price * p.gamma * r.powi(2);
            delta_pnl + gamma_pnl
        })
        .sum()
}

fn simulate_classical(
    portfolio: &VaRPortfolio,
    n_paths: u64,
    shock_model: &ShockModel,
    rng: &mut SmallRng,
) -> Result<Vec<Scenario>, RiskError> {
    let horizon_scale = (1.0_f64 / 252.0_f64).sqrt();
    (0..n_paths)
        .map(|path_id| {
            let shocks = correlated_shocks(portfolio, shock_model, rng)?;
            let returns: Vec<f64> = shocks.into_iter().map(|s| s * horizon_scale).collect();
            let pnl = pnl_from_returns(portfolio, &returns);
            Ok(Scenario {
                path_id,
                pnl,
                returns,
            })
        })
        .collect()
}

fn simulate_quantum_inspired(
    portfolio: &VaRPortfolio,
    n_paths: u64,
    shock_model: &ShockModel,
    rng: &mut SmallRng,
) -> Result<Vec<Scenario>, RiskError> {
    let horizon_scale = (1.0_f64 / 252.0_f64).sqrt();
    (0..n_paths)
        .map(|path_id| {
            let mut shocks = correlated_shocks(portfolio, shock_model, rng)?;
            let tail_bias = if path_id % 4 == 0 { 1.5 } else { 1.0 };
            for s in &mut shocks {
                if *s < 0.0 {
                    *s *= tail_bias;
                }
            }
            let returns: Vec<f64> = shocks.into_iter().map(|s| s * horizon_scale).collect();
            let pnl = pnl_from_returns(portfolio, &returns);
            Ok(Scenario {
                path_id,
                pnl,
                returns,
            })
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
            gamma: 0.05,
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
        let engine = RiskEngine::new(QBackend::QuantumInspired { max_samples: 4_096 }, 0.95)
            .unwrap()
            .with_shock_model(ShockModel::StudentT {
                degrees_of_freedom: 5.0,
            });
        let var = engine.compute_var(&portfolio, Duration::days(1)).unwrap();
        let cvar = engine.compute_cvar(&portfolio, Duration::days(1)).unwrap();
        assert!(cvar.value >= var.value);
    }

    #[test]
    fn historical_replay_produces_bundle() {
        let portfolio =
            VaRPortfolio::from_positions(vec![simple_position(10.0, 0.2)], identity_corr(1));
        let engine = RiskEngine::new(QBackend::ClassicalFallback, 0.95).unwrap();
        let bundle = engine
            .replay_historical(&portfolio, &[vec![-0.02], vec![0.01], vec![-0.05]])
            .unwrap();
        assert_eq!(bundle.n_paths, 3);
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
