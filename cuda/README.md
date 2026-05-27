# cuda notes

these files are experiments around muon-style optimizer updates. everything here assumes column-major matrices:

$$
X_{i,j} = X[i + jN]
$$

where the matrix has \(N\) rows and \(M\) columns.

## files

| file | role |
|------|------|
| `muon.cu` | baseline muon update and newton-schulz polar approximation |
| `gns_muon.cu` | gram newton-schulz variants |
| `normuon.cu` | row-ema normalized muon |
| `u_normuon.cu` | normuon with the uniform-row scaling variant |
| `aurora.cu` | practical aurora diagonal preconditioning loop |
| `riemann_aurora.cu` | riemannian aurora balanced-stiefel approximation |
| `benchmark.cu` | older gram-ns benchmark |
| `benchmark_optimizer_variants.cu` | optimizer-update benchmark for muon / normuon / u-normuon / aurora / riemann aurora |

## muon

Muon keeps a momentum matrix and applies an approximate polar update:

$$
M_t = \mu M_{t-1} + (1-\mu)G_t
$$

$$
U_t = \operatorname{polar}(M_t)
$$

The CUDA code approximates the polar factor with 5 newton-schulz iterations. First:

$$
X_0 = \frac{M_t}{\lVert M_t \rVert_F + \epsilon}
$$

then repeats:

$$
X_{k+1} = aX_k + b(X_kX_k^\top)X_k + c(X_kX_k^\top)^2X_k
$$

with the coefficients used in `muon.cu`.

The final decoupled weight-decay update is:

$$
W_{t+1} = (1-\eta\lambda)W_t - \eta U_t
$$

The transpose path in `newton_schulz_launch` keeps the expensive product on the smaller side for tall matrices.

## gram newton-schulz

`gns_muon.cu` moves the iteration onto the smaller gram matrix. For a tall matrix:

$$
G_0 = X_0^\top X_0
$$

Then it applies a matrix polynomial on \(G\):

$$
R_{k+1} = R_k \left(aI + bG_k + cG_k^2\right)
$$

and returns:

$$
U = X_0R_K
$$

The point is to avoid repeated rectangular products when \(N \gg M\). The polar and restart modes are still experimental in this repo.

## normuon

NorMuon starts from the muon polar update and normalizes rows using an ema of row energy.

For each row:

$$
v_{t,i} = \beta_2 v_{t-1,i} + (1-\beta_2)\frac{1}{M}\sum_{j=1}^{M} U_{t,i,j}^2
$$

The row-normalized update is:

$$
\hat{U}_{t,i,j} = \frac{U_{t,i,j}}{\sqrt{v_{t,i}+\epsilon}}
$$

Then the code computes:

$$
\hat{\eta} = 0.2\eta\frac{\lVert \hat{U}_t \rVert_F}{\sqrt{NM}}
$$

and applies:

$$
W_{t+1} = (1-\hat{\eta}\lambda)W_t - \hat{\eta}\hat{U}_t
$$

In the benchmark, `row_ema` is initialized to 1.0 to avoid the first step exploding on tiny rows.

## u-normuon

U-NorMuon uses the same row-ema normalization as NorMuon, but changes the final learning-rate scale:

$$
\hat{\eta} = 0.2\eta\frac{\lVert \hat{U}_t \rVert_F}{\sqrt{M}}
$$

The idea is to push toward a uniform row scale that is closer to the tall-matrix orthogonal scale.

## practical aurora

Aurora tries to balance row leverage while staying close to a polar update.

Given an update matrix \(G\), square matrices just use:

$$
U = \operatorname{polar}(G)
$$

For rectangular matrices, wide matrices are transposed first so the working matrix is tall:

$$
X =
\begin{cases}
G, & N \ge M \\
G^\top, & N < M
\end{cases}
$$

Initialize a row diagonal preconditioner:

$$
D_0 = \operatorname{diag}\left(\frac{1}{\lVert X_{1,:}\rVert_2},\ldots,\frac{1}{\lVert X_{N,:}\rVert_2}\right)
$$

The target row squared norm is:

$$
r = \frac{M}{N}
$$

Each preconditioning iteration does:

$$
U_k = \operatorname{polar}(D_kX)
$$

and updates the diagonal:

$$
D_{k+1,i} = D_{k,i}\left(\frac{r}{\lVert U_{k,i,:}\rVert_2^2}\right)^\beta
$$

Finally it applies the muon aspect-ratio scale:

$$
U \leftarrow U\sqrt{\max\left(1,\frac{N}{M}\right)}
$$

and uses the same decoupled weight-decay update.

## riemannian aurora

Riemannian Aurora targets the balanced-stiefel problem:

$$
\max_U \langle G, U\rangle
$$

subject to:

$$
U^\top U = I_M
$$

and:

$$
\lVert U_{i,:}\rVert_2^2 = \frac{M}{N}
$$

The implementation starts from:

$$
U_0 = \operatorname{polar}(G)
$$

For each outer step, compute the stiefel correction:

$$
B = \frac{1}{2}(U^\top G + G^\top U)
$$

The row-norm correction right-hand side is:

$$
q_i = \langle G_{i,:}, U_{i,:}\rangle - \langle (UB)_{i,:}, U_{i,:}\rangle
$$

with the mean removed:

$$
q \leftarrow q - \operatorname{mean}(q)
$$

The row multipliers \(\lambda\) approximately solve:

$$
\left(rI - (P \circ P)\right)\lambda = q
$$

where:

$$
P = UU^\top,\qquad r = \frac{M}{N}
$$

The CUDA code uses conjugate gradient for this solve. It does not explicitly materialize \(P \circ P\); the matvec follows the identity from the Python reference.

Then:

$$
S = B - U^\top\operatorname{diag}(\lambda)U
$$

and the tangent direction is:

$$
Z = G - US - \operatorname{diag}(\lambda)U
$$

The ascent step is:

$$
Y = U + \alpha Z
$$

Retraction alternates row normalization and polar projection:

$$
Y_{i,:} \leftarrow Y_{i,:}\frac{\sqrt{M/N}}{\lVert Y_{i,:}\rVert_2+\epsilon}
$$

$$
Y \leftarrow \operatorname{polar}(Y)
$$

Finally, like the other aurora update:

$$
U \leftarrow U\sqrt{\max\left(1,\frac{N}{M}\right)}
$$

The current CUDA defaults are lighter than the Python reference so it finishes on the laptop benchmark:

$$
\text{outer steps}=2,\quad \text{cg steps}=8,\quad \text{retraction steps}=1
$$

## benchmark metrics

`benchmark_optimizer_variants.cu` uses synthetic gradients with row anisotropy:

$$
G_{i,j}=s_i z_{i,j},\qquad z_{i,j}\sim\mathcal{N}(0,1)
$$

with about 20 percent of rows using:

$$
s_i = 10^{-3}
$$

and the rest using:

$$
s_i = 1
$$

The reported metrics are:

$$
\text{row cv} = \frac{\operatorname{std}_i(\lVert U_{i,:}\rVert_2)}{\operatorname{mean}_i(\lVert U_{i,:}\rVert_2)}
$$

which tests update mass uniformity / neuron death risk.

$$
\text{dead row fraction} = \frac{\#\{i:\lVert U_{i,:}\rVert_2 < 0.01\operatorname{mean}_j\lVert U_{j,:}\rVert_2\}}{N}
$$

which catches rows receiving almost no update.

For \(N \ge M\):

$$
\text{orthogonality defect} =
\frac{\lVert U^\top U - I\rVert_F}{\lVert I\rVert_F}
$$

and for \(N < M\):

$$
\text{orthogonality defect} =
\frac{\lVert UU^\top - I\rVert_F}{\lVert I\rVert_F}
$$

which tests whether the muon polar geometry survived the extra balancing.

Gradient alignment is:

$$
\frac{\langle G,U\rangle}{\lVert G\rVert_F\lVert U\rVert_F}
$$

which checks whether the update still points with the synthetic gradient.

Runtime is the average CUDA event time around the optimizer step only:

$$
\text{avg step ms}
$$
