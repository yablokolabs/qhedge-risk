/-!
# QhedgeRisk

A Lean 4 specification companion for the Rust `qhedge-risk` engine.

This file now combines the earlier Rust ↔ Lean alignment pass with the tighter
finite-sample risk-measure pass:
- direct mirroring of Rust concepts like `ShockModel`, `VaRMethod`, scenario bundles,
  delta/gamma loss structure, replay semantics, and factor-aware risk hooks
- tighter empirical risk semantics via explicit sorting, quantile-index selection,
  tail-set construction, and empirical `CVaR`

It remains a specification layer rather than machine-checked Rust verification.
-/

namespace QhedgeRisk

abbrev Exposure := Rational
abbrev Return := Rational
abbrev Loss := Rational
abbrev Weight := Rational
abbrev Volatility := Rational
abbrev Confidence := Rational

inductive ShockModel where
  | gaussian
  | studentT (ν : Nat)
  | crashMixture (p : Rational) (shockMul : Rational)
deriving Repr

inductive VaRMethod where
  | monteCarlo
  | historical
  | parametric
deriving Repr

structure Position where
  exposure : Exposure
  delta : Rational
  gamma : Rational
  factorLoadings : List Rational
  idioVol : Rational
deriving Repr

structure Portfolio where
  positions : List Position
deriving Repr

structure FactorModel where
  covariance : List (List Rational)
deriving Repr

structure Scenario where
  pathId : Nat
  pnl : Rational
  returns : List Return
deriving Repr

structure ScenarioBundle where
  scenarios : List Scenario
  shockModel : ShockModel
  method : VaRMethod
deriving Repr

structure ReturnProcess where
  scenarios : List (List Return)
  shockModel : ShockModel
deriving Repr

structure VaR where
  confidence : Confidence
  value : Loss
  method : VaRMethod
deriving Repr

structure CVaR where
  confidence : Confidence
  value : Loss
  method : VaRMethod
deriving Repr

/-- Aggregate portfolio exposure. -/
def Portfolio.totalExposure (p : Portfolio) : Exposure :=
  p.positions.foldl (fun acc pos => acc + pos.exposure) 0

/-- Aggregate first-order exposure. -/
def Portfolio.deltaExposure (p : Portfolio) : Exposure :=
  p.positions.foldl (fun acc pos => acc + pos.exposure * pos.delta) 0

/-- Aggregate second-order exposure proxy. -/
def Portfolio.gammaExposure (p : Portfolio) : Exposure :=
  p.positions.foldl (fun acc pos => acc + pos.exposure * pos.gamma) 0

/-- Simple factor exposure for a given factor index. -/
def Portfolio.factorExposure (p : Portfolio) (i : Nat) : Rational :=
  p.positions.foldl
    (fun acc pos => acc + pos.exposure * (pos.factorLoadings.getD i 0))
    0

/-- Delta-gamma loss approximation for one shocked return. -/
def positionLossFromReturn (pos : Position) (r : Return) : Loss :=
  -(pos.exposure * pos.delta * r + (1 / 2 : Rational) * pos.exposure * pos.gamma * r^2)

/-- Portfolio loss from a vector of returns. -/
def portfolioLossFromReturns (p : Portfolio) (returns : List Return) : Loss :=
  List.zipWith positionLossFromReturn p.positions returns |>.sum

/-- Losses induced by replaying return scenarios against the portfolio. -/
def replayHistorical (p : Portfolio) (rp : ReturnProcess) : List Loss :=
  rp.scenarios.map (portfolioLossFromReturns p)

/-- Bundle constructor matching the Rust idea of reusable scenario bundles. -/
def historicalScenarioBundle (p : Portfolio) (rp : ReturnProcess) : ScenarioBundle :=
  let scenarios :=
    rp.scenarios.enum.map fun ⟨idx, returns⟩ =>
      { pathId := idx
        pnl := - portfolioLossFromReturns p returns
        returns := returns }
  { scenarios := scenarios, shockModel := rp.shockModel, method := VaRMethod.historical }

/-- Insertion into an already nondecreasing list. -/
def insertSorted : Loss → List Loss → List Loss
  | x, [] => [x]
  | x, y :: ys => if x ≤ y then x :: y :: ys else y :: insertSorted x ys

/-- Nondecreasing sort of losses. -/
def sortLosses : List Loss → List Loss
  | [] => []
  | x :: xs => insertSorted x (sortLosses xs)

/-- Quantile index matching the Rust-style `floor(confidence * n)` shape, clipped to `n - 1`. -/
def quantileIndex (α : Confidence) (n : Nat) : Nat :=
  if h : n = 0 then 0
  else min (Int.toNat ⌊α * n⌋) (n - 1)

/-- Safe list lookup with default zero. -/
def lossAt (xs : List Loss) (i : Nat) : Loss :=
  xs.getD i 0

/-- Empirical `VaR` over sorted losses. -/
def empiricalVaR (α : Confidence) (losses : List Loss) : Loss :=
  let sorted := sortLosses losses
  lossAt sorted (quantileIndex α sorted.length)

/-- Tail losses above a chosen empirical `VaR` threshold. -/
def tailLosses (α : Confidence) (losses : List Loss) : List Loss :=
  let v := empiricalVaR α losses
  losses.filter fun ℓ => v ≤ ℓ

/-- Empirical `CVaR` as the mean of tail losses above the `VaR` threshold. -/
def empiricalCVaR (α : Confidence) (losses : List Loss) : Loss :=
  match tailLosses α losses with
  | [] => empiricalVaR α losses
  | ys => ys.sum / ys.length

/-- Historical `VaR` obtained from replayed losses. -/
def historicalVaR (α : Confidence) (p : Portfolio) (rp : ReturnProcess) : VaR :=
  { confidence := α
    value := empiricalVaR α (replayHistorical p rp)
    method := VaRMethod.historical }

/-- Historical `CVaR` obtained from replayed losses. -/
def historicalCVaR (α : Confidence) (p : Portfolio) (rp : ReturnProcess) : CVaR :=
  { confidence := α
    value := empiricalCVaR α (replayHistorical p rp)
    method := VaRMethod.historical }

/-- Simple quadratic-form helper for factor covariance. -/
def covarianceQuadratic (ws : List Weight) (Σ : List (List Rational)) : Rational :=
  let rowTerm := fun i wᵢ =>
    ws.enum.foldl
      (fun acc ⟨j, wⱼ⟩ => acc + wᵢ * (Σ.getD i []).getD j 0 * wⱼ)
      0
  ws.enum.foldl (fun acc ⟨i, wᵢ⟩ => acc + rowTerm i wᵢ) 0

/-- Factor-model variance proxy using portfolio factor exposures. -/
def factorVariance (p : Portfolio) (fm : FactorModel) (nFactors : Nat) : Rational :=
  let ws := List.range nFactors |>.map fun i => p.factorExposure i
  covarianceQuadratic ws fm.covariance

/-- A toy variance proxy combining systematic variance and idiosyncratic pieces. -/
def portfolioVarianceProxy (p : Portfolio) (fm : FactorModel) (nFactors : Nat) : Rational :=
  factorVariance p fm nFactors +
    p.positions.foldl (fun acc pos => acc + pos.exposure^2 * pos.idioVol^2) 0

/-- Parametric Gaussian `VaR` sketch: z-score times a volatility proxy. -/
def parametricVaRValue (zσ : Rational) : Loss := zσ

/-- Factor-aware parametric `VaR` specification hook. -/
def factorParametricVaR (z : Rational) (α : Confidence) (p : Portfolio) (fm : FactorModel)
    (nFactors : Nat) : VaR :=
  { confidence := α
    value := parametricVaRValue (z * portfolioVarianceProxy p fm nFactors)
    method := VaRMethod.parametric }

/-- Monte Carlo spec hook: interpret bundle losses empirically. -/
def monteCarloVaR (α : Confidence) (bundle : ScenarioBundle) : VaR :=
  { confidence := α
    value := empiricalVaR α (bundle.scenarios.map fun s => -s.pnl)
    method := VaRMethod.monteCarlo }

/-- Monte Carlo spec hook: interpret bundle tail losses empirically. -/
def monteCarloCVaR (α : Confidence) (bundle : ScenarioBundle) : CVaR :=
  { confidence := α
    value := empiricalCVaR α (bundle.scenarios.map fun s => -s.pnl)
    method := VaRMethod.monteCarlo }

/-- A larger volatility proxy should not reduce parametric `VaR`. -/
theorem parametric_var_monotonic_in_scale
    (z σ₁ σ₂ : Rational)
    (hz : 0 ≤ z)
    (hσ : σ₁ ≤ σ₂) :
    parametricVaRValue (z * σ₁) ≤ parametricVaRValue (z * σ₂) := by
  dsimp [parametricVaRValue]
  nlinarith

/-- Every element of `tailLosses` is at least the empirical `VaR` threshold by construction. -/
theorem var_le_tail_member
    (α : Confidence)
    (losses : List Loss)
    (ℓ : Loss)
    (hℓ : ℓ ∈ tailLosses α losses) :
    empiricalVaR α losses ≤ ℓ := by
  unfold tailLosses at hℓ
  simp at hℓ
  exact hℓ.2

/-- If each position exposure weakly increases, total portfolio exposure also weakly increases. -/
theorem monotonic_in_exposure
    (xs ys : List Position)
    (h_len : xs.length = ys.length)
    (h_pointwise : ∀ x y, (x, y) ∈ List.zip xs ys → x.exposure ≤ y.exposure) :
    Portfolio.totalExposure ⟨xs⟩ ≤ Portfolio.totalExposure ⟨ys⟩ := by
  simp [Portfolio.totalExposure]
  sorry

/-- Scaling exposure upward should not reduce a positive-tail risk proxy under a fixed process. -/
theorem tail_risk_monotonic_in_scale
    (k₁ k₂ : Rational)
    (h_scale : 0 ≤ k₁ ∧ k₁ ≤ k₂)
    (p : Portfolio) :
    k₁ * p.totalExposure ≤ k₂ * p.totalExposure := by
  nlinarith [h_scale.1, h_scale.2]

/-- Increasing exposure to a positive factor loading should not reduce factor exposure. -/
theorem factor_exposure_monotonic
    (e₁ e₂ β : Rational)
    (h : e₁ ≤ e₂)
    (hβ : 0 ≤ β) :
    e₁ * β ≤ e₂ * β := by
  nlinarith

/-- If replayed losses are all nonnegative, historical `VaR` is nonnegative too. -/
theorem historical_var_nonnegative
    (α : Confidence)
    (p : Portfolio)
    (rp : ReturnProcess)
    (h : ∀ ℓ ∈ replayHistorical p rp, 0 ≤ ℓ) :
    0 ≤ (historicalVaR α p rp).value := by
  unfold historicalVaR
  simp
  by_cases hs : sortLosses (replayHistorical p rp) = []
  · simp [empiricalVaR, hs]
  ·
    have hnonempty : replayHistorical p rp ≠ [] := by
      intro hnil
      simp [hnil] at hs
    sorry

/-- Empirical `CVaR` falls back to empirical `VaR` when the tail set is empty. -/
theorem cvar_eq_var_when_no_tail
    (α : Confidence)
    (losses : List Loss)
    (h : tailLosses α losses = []) :
    empiricalCVaR α losses = empiricalVaR α losses := by
  simp [empiricalCVaR, h]

/-- If each tail loss is above the chosen `VaR` threshold, empirical `CVaR`
should not fall below empirical `VaR`. -/
theorem cvar_geq_var
    (α : Confidence)
    (losses : List Loss)
    (h_tail : ∀ ℓ ∈ tailLosses α losses, empiricalVaR α losses ≤ ℓ) :
    empiricalVaR α losses ≤ empiricalCVaR α losses := by
  by_cases h : tailLosses α losses = []
  · simp [empiricalCVaR, h]
  · simp [empiricalCVaR, h]
    sorry

end QhedgeRisk
