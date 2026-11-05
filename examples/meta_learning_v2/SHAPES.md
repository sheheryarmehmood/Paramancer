# Meta-learning shape map

Keep this notation nearby when following tensors through an episode:

| Symbol | Meaning |
|---|---|
| `B` | Number of episodes in the meta-batch |
| `S` | Support examples per episode |
| `Q` | Query examples per episode |
| `C` | Episode-local classes (`n_way`) |
| `D` | Encoder output / embedding dimension |
| `Cin, H, W` | Image channels, height, and width |

## Data flow

| Value | Shape |
|---|---|
| `support_images` | `(B, S, Cin, H, W)` |
| `support_targets` | `(B, S)` with labels in `0, ..., C - 1` |
| `support_features` | `(B, S, D)` |
| `query_images` | `(B, Q, Cin, H, W)` |
| `query_targets` | `(B, Q)` with labels in `0, ..., C - 1` |
| `query_features` | `(B, Q, D)` |
| classifier weight `W` | `(B, C, D)` |
| classifier bias `b` | `(B, C)` |
| support logits | `(B, S, C)` |
| query logits | `(B, Q, C)` |

The prediction performed by `BaseLearner` is:

```text
logits[b, sample, class]
    = dot(features[b, sample], W[b, class]) + b[b, class]
```

Thus each episode owns an independent `(W, b)` problem; batching only runs
those problems together with tensor operations.

## Warm-start states

- Forward initialization and `wb_opt` are both `(W, b)` with the shapes above.
- GD has one adjoint state shaped like `(W, b)`.
- HB, NAG, and FISTA use a lifted adjoint state
  `((W_current, b_current), (W_previous, b_previous))`.
- PDHG has its own primal-dual pair and is not used by this classifier example.
