# Project To-Do List (Postponed Tasks)

This file tracks features and improvements postponed for later stages of development in the `optim` module.

---

## Implementation & Code Design

- [ ] Add support for **callbacks or error loggers** in `Optimizer`.
- [x] Store **iteration history** (e.g., `x_curr`, metric values).
- [ ] Handle **`x_true` comparison** inside custom `metric` function.
- [x] Use consistent `__call__()` vs `step()` interface across `OptimizerStep` implementations.
- [ ] Possibly introduce a **registry pattern** for steps/schedulers to simplify instantiation and configuration.
- [ ] Review/refactor **`MomentumStep`** if it becomes more widely used or more complex.
- [ ] Allow the optimization variable to be `tuple[torch.Tensor]` in `Optimizer` and all the children classes of `OptimizerStep` as well.
- [x] Implement `PDHGStep`.
- [ ] Incorporate `tuple[torch.Tensor]` in `OptimizerStep` and `Optimizer`.
- [ ] Incorporate `PDHGStep` in `Optimizer` and implement `class PDHG(Optimizer): ...`.
- [ ] 

---

## Testing & Validation

- [x] Add **tests** for each `OptimizerStep` child and the `Optimizer` class.
- [ ] Test `PDHGStep` and `PDHG`.
- [ ] Create **example problems** (e.g., least squares, logistic regression) to test optimizers in practice.
- [ ] Add test coverage for **line search vs fixed scheduler** behavior.

---

## Project Organization

- [ ] Split `step.py` into **dedicated modules** like `gd.py`, `momentum.py`, etc.
- [ ] Add proper **`__init__.py` exports** for public-facing API (some already done).
- [ ] Add a **`README.md`** with installation and usage instructions.
- [ ] Prepare for packaging (`setup.py` / `pyproject.toml`) if publishing on PyPI or elsewhere.

---

## API & UX Enhancements

- [ ] Add `__repr__()` or `__str__()` methods to key classes for **clearer logs/debugging**.
- [ ] Possibly **standardize `grad_map` and `metric`** interfaces using Protocols or base classes.

> **Note on standardizing `grad_map` and `metric`:**  
> Right now, these are plain `callable`s (functions or lambdas). In the future, you might benefit from creating a lightweight interface or Protocol (e.g., `GradMapProtocol`, `MetricProtocol`) to enforce structure, document expectations (like required input/output signatures), and add extra features like tracking evaluations, caching, or batch handling. This is optional and only becomes useful when the project scales.
