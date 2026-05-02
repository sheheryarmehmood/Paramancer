# Project To-Do List (Postponed Tasks)

## `bloptim` Module

* [x] Implement `bloptim.implicit.optimizer.Optimizer` and its children.
* [x] Implement `bloptim.unrolled.optimizer.Optimizer` and its children.
* [ ] Implement a warm-starting wrapper (forward-only for the unrolled Optimizers and forward and backward for the implicit Optimizers)
* [ ] Implement Bilevel optimizer which receives a ...

## Applications

* [ ] Meta Learning (Bertinetto et al., ICLR 2019).
* [ ] monDEQ (Winston and Kolter, NeurIPS 2020).

## Clean-ups and Refactors

* [ ] Perform a clean-up on `variable.types`.
* [ ] Refactor `operators.grad.gradient` and `operators.linalg.adjoint`.
