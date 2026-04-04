# Quantile Alignment Notes

This PR tightens the Lean side around the empirical `VaR` / `CVaR` story used by the Rust engine.

## What got closer

### Rust
Rust computes empirical `VaR` by:
1. turning scenario `PnL` into losses
2. sorting the losses
3. picking `floor(confidence * n)`
4. clipping to the last valid index

Rust computes empirical `CVaR` by:
1. computing empirical `VaR`
2. selecting losses greater than or equal to that threshold
3. averaging that tail set

### Lean
Lean now mirrors those ingredients explicitly via:
- `sortLosses`
- `quantileIndex`
- `lossAt`
- `empiricalVaR`
- `tailLosses`
- `empiricalCVaR`

## Remaining gap
The Lean side still does not prove that `sortLosses` is ordered or that the quantile semantics are fully identical to the Rust implementation for all edge cases.

So this is:
- **much tighter than before**
- **closer to 1:1 finite-sample semantics**
- **still not a formal proof of algorithmic equivalence**
