# optimizer methods

this note collects the optimizer variants used in this repo and writes down the update rules in one place. it is meant to be a technical map of the experiments, not a claim that any one method is globally best.

notation:

- $$W_t$$ is the parameter matrix.
- $$G_t = \nabla_W \ell_t(W_t)$$ is the current gradient.
- $$\eta$$ is the learning rate.
- $$\lambda$$ is decoupled weight decay.
- $$\mathrm{polar}(X)$$ denotes the orthogonal / polar-like factor computed here with newton-schulz iterations.
- $$\epsilon$$ is a small numerical stabilizer.

## baseline muon

Muon first builds a momentum matrix and then applies an approximate polar update.

$$
M_t = \mu M_{t-1} + (1-\mu)G_t
$$

with nesterov enabled, the matrix sent to the polar map is:

$$
\widetilde M_t = (1-\mu)G_t + \mu M_t
$$

then:

$$
U_t = \mathrm{polar}(\widetilde M_t)
$$

and the parameter update is:

$$
W_{t+1} = (1-\eta\lambda)W_t - \eta s(W_t) U_t
$$

where this repo usually uses the shape factor:

$$
s(W) = \sqrt{\max\left(1, \frac{m}{n}\right)}
$$

for an $$m \times n$$ matrix. the main idea is simple: keep the update direction close to a polar factor, so the step has controlled matrix geometry rather than raw gradient scale.

## normuon

NorMuon starts from the same polar update as Muon, then normalizes rows by a running estimate of row energy.

given:

$$
U_t = \mathrm{polar}(\widetilde M_t)
$$

compute per-row mean squared update energy:

$$
r_{t,i} = \frac{1}{n}\sum_{j=1}^{n} U_{t,ij}^2
$$

track an exponential moving average:

$$
R_{t,i} = \beta_2 R_{t-1,i} + (1-\beta_2)r_{t,i}
$$

then row-normalize:

$$
\widehat U_{t,ij} = \frac{U_{t,ij}}{\sqrt{R_{t,i}}+\epsilon}
$$

The repo variant then uses a norm-adjusted learning rate:

$$
\widehat\eta_t = 0.2\eta\frac{\|\widehat U_t\|_F}{\sqrt{mn}}
$$

and applies:

$$
W_{t+1} = (1-\eta\lambda)W_t - \widehat\eta_t \widehat U_t
$$

The point is to reduce row-energy collapse without leaving the Muon-style polar update family.

## u-normuon

U-NorMuon uses the same row-normalized update as NorMuon but changes the learning-rate normalization:

$$
\widehat\eta_t = 0.2\eta\frac{\|\widehat U_t\|_F}{\sqrt{n}}
$$

instead of normalizing by $$\sqrt{mn}$$. this makes the step scale less aggressively with the number of rows. in the current synthetic benchmark, this is one reason `u_normuon` often looks stronger on rectangular cases.

## aurora

Aurora is an attempt to balance row leverage while staying close to a polar-style update.

for square matrices, the current implementation mostly falls back to:

$$
U_t = \mathrm{polar}(\widetilde M_t)
$$

for rectangular matrices, it works on an orientation where rows are the smaller / balanced dimension. let:

$$
X = \widetilde M_t
$$

and initialize a positive diagonal row-scaling vector $$d_0$$. for iteration $$k$$:

$$
U_k = \mathrm{polar}(d_k \odot X)
$$

where $$d_k \odot X$$ means row-wise scaling. then compute row squared norms:

$$
q_{k,i} = \sum_j U_{k,ij}^2
$$

and update the diagonal scale toward a target row energy:

$$
d_{k+1,i} = d_{k,i}\left(\frac{q^\star}{q_{k,i}+\epsilon}\right)^\alpha
$$

where $$q^\star$$ is the target row squared norm and $$\alpha$$ is a small balancing exponent.

the resulting update is then used like a Muon update:

$$
W_{t+1} = (1-\eta\lambda)W_t - \eta s(W_t) U_t
$$

The synthetic benchmark shows the tradeoff pretty clearly: Aurora can improve some dead-row / direction cases, but on this grid it is not a general winner.

## riemannian aurora

Riemannian Aurora pushes the Aurora idea harder: instead of only applying a few diagonal balancing steps, it treats the balanced update as a constrained geometry problem.

informally, the target is an update matrix $$U$$ that is close to the gradient direction while satisfying row-balance and polar/Stiefel-like constraints:

$$
\min_U \|U - X\|_F^2
$$

subject to approximate constraints of the form:

$$
UU^\top \approx I
$$

and:

$$
\|U_{i,:}\|_2^2 \approx q^\star
$$

The CUDA implementation uses a practical approximation with alternating projection / retraction-style steps. one part pulls the update toward the polar manifold:

$$
Y \leftarrow \mathrm{polar}(Y)
$$

and another part tries to restore row-balance:

$$
Y_{i,:} \leftarrow \sqrt{q^\star}\frac{Y_{i,:}}{\|Y_{i,:}\|_2+\epsilon}
$$

This gives strong row-balance metrics in the synthetic benchmark, but the current implementation is expensive. so the repo treats it as a useful reference point for geometry, not as a cheap training optimizer.

## schedule-free averaging

The schedule-free wrappers keep three parameter sequences:

- $$z_t$$: the sequence being directly updated by the optimizer.
- $$x_t$$: the averaged sequence used for evaluation.
- $$y_t$$: the interpolation used for training.

Training uses:

$$
y_t = (1-\beta_t)z_t + \beta_t x_t
$$

After an optimizer step updates $$z_t$$, the average is updated with an $$\eta_t^2$$ weight:

$$
c_t = c_{t-1} + \eta_t^2
$$

$$
x_t = \left(1-\frac{\eta_t^2}{c_t}\right)x_{t-1}
      + \frac{\eta_t^2}{c_t}z_t
$$

This is why the implementation tracks $$c_t$$ directly instead of using a plain arithmetic average.

The beta schedule is anchored at the warmup boundary $$T_0$$:

$$
c_{T_0} = c_t \quad \text{at the end of warmup}
$$

for $$t > T_0$$:

$$
\beta_t = 1 - \left(\frac{c_{T_0}}{c_t}\right)^\rho(1-\beta_1)
$$

during warmup:

$$
\beta_t = \beta_1
$$

This is the schedule-free rule used by the AMUSE and SF-AdamW wrappers in this repo.

## schedule-free adamw

The schedule-free AdamW baseline applies AdamW to the $$z_t$$ sequence, then uses the schedule-free averaging rule above for $$x_t$$.

Adam moments:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)G_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)G_t^2
$$

bias-corrected update:

$$
\widehat m_t = \frac{m_t}{1-\beta_1^t}
$$

$$
\widehat v_t = \frac{v_t}{1-\beta_2^t}
$$

$$
z_{t+1} = (1-\eta\lambda)z_t
          - \eta \frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}
$$

then:

$$
x_{t+1} =
\left(1-\frac{\eta^2}{c_{t+1}}\right)x_t
+ \frac{\eta^2}{c_{t+1}}z_{t+1}
$$

This baseline is useful because it separates schedule-free averaging behavior from Muon-specific matrix geometry.

## amuse-muon

AMUSE-Muon combines schedule-free averaging with a Muon-like matrix update. the important implementation detail in this repo is that the matrix path does not use Adam's first moment. it keeps the second moment only.

For matrix parameters:

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)G_t^2
$$

with bias correction:

$$
\widehat v_t = \frac{v_t}{1-\beta_2^t}
$$

precondition the gradient:

$$
P_t = \frac{G_t}{\sqrt{\widehat v_t}+\epsilon}
$$

then take a Muon-style polar update:

$$
U_t = \mathrm{polar}(P_t)
$$

and apply the usual rectangular shape factor:

$$
U_t \leftarrow s(W_t)U_t
$$

The d-muon-style scale matching step rescales the polar update to match the RMS of the preconditioned gradient:

$$
\mathrm{rms}(A) = \frac{\|A\|_F}{\sqrt{\mathrm{numel}(A)}}
$$

$$
U_t \leftarrow U_t
\frac{\mathrm{rms}(P_t)}{\mathrm{rms}(U_t)+\epsilon}
$$

then update:

$$
z_{t+1} = (1-\eta\lambda)z_t - \eta U_t
$$

and use the same schedule-free averaging rule:

$$
c_{t+1} = c_t + \eta^2
$$

$$
x_{t+1} =
\left(1-\frac{\eta^2}{c_{t+1}}\right)x_t
+ \frac{\eta^2}{c_{t+1}}z_{t+1}
$$

For non-matrix parameters, the wrapper falls back to AdamW-style updates on $$z_t$$, then uses the same $$x_t$$ averaging.

The current character-lm run shows a useful distinction: AMUSE reaches the best single validation point, but the best AMUSE settings drift upward later in training. that means the method is strong on peak validation in this sweep, while `sf_adamw` is stronger on late-run stability.

## how to read the plots

The loss plots in `artifacts/char_lm/` and `artifacts/char_lm_amuse_fix/` show raw validation loss over time, not best-so-far loss. that matters for the AMUSE run:

$$
\min_t \mathrm{val}(t)
$$

and:

$$
\mathrm{val}(T)
$$

answer different questions. AMUSE has the best value of $$\min_t \mathrm{val}(t)$$ in the current sweep, while `sf_adamw` has the cleaner late-run value near $$T = 15000$$.
