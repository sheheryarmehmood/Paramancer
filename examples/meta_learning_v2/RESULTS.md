# Experiment Results

## Configuration space

Let

\[
\begin{aligned}
\mathcal B &= \{\text{unrolled},\ \text{implicit}\},\\
\mathcal A &\subseteq \{\mathrm{GD},\ \mathrm{HB},\ \mathrm{NAG}\},\\
\mathcal I &\subseteq \{\text{cold},\ \text{forward-warm}\},\\
\mathcal T &\subseteq \{0,1,\ldots,K\},\\
\mathcal S &\subseteq \mathbb N_0,
\end{aligned}
\]

where

\[
K=300,\qquad \omega=2,\qquad E=5.
\]

Truncation applies only to unrolling. Hence the actual sweep is

\[
\mathcal C=
\bigl(\{\text{unrolled}\}\times\mathcal A\times\mathcal I
\times\mathcal T\times\mathcal S\bigr)
\;\cup\;
\bigl(\{\text{implicit}\}\times\mathcal A\times\mathcal I
\times\{0\}\times\mathcal S\bigr).
\]

Therefore,

\[
|\mathcal C|
=|\mathcal A|\,|\mathcal I|\,|\mathcal S|
\bigl(|\mathcal T|+1\bigr).
\]

The command-line defaults are

\[
\mathcal A=\{\mathrm{GD}\},\quad
\mathcal I=\{\text{cold}\},\quad
\mathcal T=\{0\},\quad
\mathcal S=\{0\},
\]

and thus produce two configurations.

## Fixed experiment parameters

\[
\begin{array}{c|c}
\text{quantity} & \text{value}\\ \hline
\text{epochs }E & 5\\
\text{episodes per epoch} & 20\\
\text{meta-batch size} & 2\\
N\text{-way} & 5\\
\text{support examples per class} & 1\\
\text{query examples per class} & 2\\
\text{embedding dimension} & 64\\
\text{inner iterations }K & 300\\
\text{inner tolerance} & 10^{-5}\\
\lambda & 10^{-2}\\
\text{outer learning rate} & 10^{-3}
\end{array}
\]

## Reference

For every episode and current encoder parameter \(\theta\), the reference uses

\[
w_{\mathrm{ref}}
\approx \arg\min_w L_{\mathrm{support}}(w,\theta)
\]

with HB, implicit differentiation, at most \(2000\) iterations, and tolerance
\(10^{-10}\). Define

\[
J_{\mathrm{ref}}(\theta)
=L_{\mathrm{query}}(w_{\mathrm{ref}},\theta),
\qquad
g_{\mathrm{ref}}
=\nabla_\theta J_{\mathrm{ref}}(\theta).
\]

The candidate and reference reuse the same encoded support and query features.

## Recorded results

Let \(w_0\), \(w_{\mathrm{idle}}\), and \(w_K\) denote the initial,
post-idle, and final lower-level parameters. For implicit differentiation and
full unrolling,

\[
w_{\mathrm{idle}}=w_0.
\]

The recorded forward errors are

\[
\begin{aligned}
e_0 &= |J(w_0,\theta)-J_{\mathrm{ref}}(\theta)|,\\
e_{\mathrm{idle}}
&=|J(w_{\mathrm{idle}},\theta)-J_{\mathrm{ref}}(\theta)|,\\
e_K &= |J(w_K,\theta)-J_{\mathrm{ref}}(\theta)|.
\end{aligned}
\]

The recorded reverse error is

\[
e_{\mathrm{rev}}
=\frac{\|g_K-g_{\mathrm{ref}}\|_2}
       {\|g_{\mathrm{ref}}\|_2},
\qquad
g_K=\nabla_\theta J(w_K,\theta).
\]

Each epoch produces

\[
\left(
J_K,
\operatorname{accuracy},
e_0,
e_{\mathrm{idle}},
e_K,
e_{\mathrm{rev}},
t_{\mathrm{fwd}},
t_{\mathrm{bwd}}
\right).
\]

For \(e_0,e_{\mathrm{idle}},e_K,e_{\mathrm{rev}},t_{\mathrm{fwd}}\), and
\(t_{\mathrm{bwd}}\), the per-run CSV stores

\[
\operatorname{mean}_{\text{meta-batches}}(\cdot),
\qquad
\operatorname{std}_{\text{meta-batches}}(\cdot),
\]

where the standard deviation is the population standard deviation. Loss is
the query-weighted epoch mean and accuracy is the epoch query accuracy.

Here

\[
\begin{aligned}
t_{\mathrm{fwd}}
&=t(\text{encoding}+\text{inner solve}+\text{query loss}),\\
t_{\mathrm{bwd}}
&=t(\texttt{loss.backward()}).
\end{aligned}
\]

Reference computation and data transfer are excluded from both timings. CUDA
is synchronized at timing boundaries.

## CSV output

One file is written for every \(c\in\mathcal C\):

```text
backend=<B>_algorithm=<A>_initialization=<I>_truncation=<T>_seed=<S>.csv
```

Each row represents one epoch. The per-run files are combined into

```text
results/summary.csv
```

The per-run files contain only the epoch index and measured quantities. The
summary additionally restores backend, algorithm, initialization, truncation,
and seed from each filename. Cross-seed means, cross-seed standard deviations,
tables, and plots are derived from the summary after the sweep.
