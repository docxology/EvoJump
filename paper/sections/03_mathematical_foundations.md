# Mathematical Foundations

## Stochastic Process Modeling in Biology

Stochastic differential equations (SDEs) model biological processes with deterministic trends and random fluctuations (Lande 1976, Turelli 1977). The general jump-diffusion form is:

$$dX_t = \mu(X_t, t)dt + \sigma(X_t, t)dW_t + dJ_t \label{eq:jump_diffusion}$$

This equation captures three components of phenotypic change:

- **Deterministic drift** ($\mu(X_t, t)dt$): Expected directional change from developmental programs or selection
- **Continuous variation** ($\sigma(X_t, t)dW_t$): Stochastic fluctuations from noise, environment, or genetics, scaling as $\sqrt{dt}$
- **Discontinuous jumps** ($dJ_t$): Sudden discrete transitions (metamorphosis, environmental shifts)

Specialized forms address specific biological scenarios:

### Ornstein-Uhlenbeck Processes

For homeostatic traits (body temperature, metabolic rates):

$$dX_t = \kappa(\theta - X_t)dt + \sigma dW_t \label{eq:ou_process}$$

This introduces **mean reversion**: drift term $\kappa(\theta - X_t)$ pulls traits toward equilibrium $\theta$ at rate $\kappa$, modeling homeostatic regulation and stabilizing selection.

### Fractional Brownian Motion

For processes with long-range temporal dependencies (epigenetic inheritance, developmental constraints):

$$X_t = X_0 + \int_0^t f(t, s) dW_s$$

Exhibits temporal correlations via Hurst parameter $H \in (0,1)$:
- $H > 0.5$: Persistent (momentum)
- $H < 0.5$: Anti-persistent (oscillation)
- $H = 0.5$: Standard Brownian motion (independent increments)

Models situations where early developmental events create lasting biases.

### Cox-Ingersoll-Ross Process

For non-negative traits with state-dependent volatility (e.g., cell counts, gene expression levels, resource allocation):

$$dX_t = \kappa(\theta - X_t)dt + \sigma\sqrt{X_t} dW_t \label{eq:cir_intro}$$

Combines mean reversion with **square-root diffusion** $\sigma\sqrt{X_t}$ for:
- Non-negativity (noise vanishes as $X_t \to 0$)
- State-dependent volatility scaling with $\sqrt{X_t}$
- Reflecting boundary at zero without artificial constraints

Models biological phenomena where variability scales with population size or concentration.

### Lévy Processes

For heavy-tailed processes (population catastrophes, large-effect mutations):

$$dX_t = \mu dt + \sigma dW_t + dL_t^\alpha \label{eq:levy_process}$$

$L_t^\alpha$ is $\alpha$-stable Lévy process with $\alpha \in (0, 2]$:
- $\alpha = 2$: Gaussian process
- $\alpha < 2$: Heavy tails—extreme events more frequent than Gaussian models predict

Captures developmental systems where rare large jumps shape phenotypic distributions and evolutionary dynamics.

## General Framework

Let $(X_t)_{t \geq 0}$ be a stochastic process describing developmental trajectories of phenotypic traits. Mathematically, $X_t$ evolves on a filtered probability space $(\Omega, \mathcal{F}, (\mathcal{F}_t)_{t \geq 0}, \mathbb{P})$ where $\Omega$ contains all possible outcomes, $\mathcal{F}$ collects observable events, $(\mathcal{F}_t)_{t \geq 0}$ captures information flow over time, and $\mathbb{P}$ assigns probabilities.

### Jump-Diffusion Framework

The general jump-diffusion model unifies continuous and discontinuous dynamics:

$$dX_t = \mu(X_t, t)dt + \sigma(X_t, t)dW_t + \int_{\mathbb{R}} z \tilde{N}(dt, dz) \label{eq:general_jump_diffusion}$$

Components:
- **Brownian motion** ($W_t$): Continuous-time random walk with independent Gaussian increments
- **Compensated Poisson measure** ($\tilde{N}(dt, dz)$): Framework for jumps at random times with random magnitudes $z$
- **Drift function** ($\mu$): Deterministic expected change from developmental programs, selection, or environmental trends
- **Diffusion coefficient** ($\sigma$): Magnitude of continuous stochastic fluctuations, can depend on state and time

## Ornstein-Uhlenbeck Process with Jumps

### Model Specification

The Ornstein-Uhlenbeck (OU) process with Poisson jumps combines mean-reverting continuous dynamics with discrete transitions, making it particularly suitable for modeling homeostatic regulation punctuated by developmental transitions:

$$dX_t = \kappa(\theta - X_t)dt + \sigma dW_t + dJ_t \label{eq:ou_jump}$$

The biological interpretation of each parameter is crucial for application:

- $\kappa > 0$ is the **mean reversion speed**—quantifying how quickly the trait returns to equilibrium after perturbation. Larger $\kappa$ indicates stronger homeostatic regulation. The characteristic timescale of regulation is $1/\kappa$: after time $1/\kappa$, approximately 63% of a deviation has been corrected.

- $\theta$ is the **long-term equilibrium level**—the target value toward which regulation drives the trait. In evolutionary terms, this represents the fitness optimum under stabilizing selection. In physiological terms, it is the homeostatic setpoint.

- $\sigma > 0$ is the **diffusion coefficient**—measuring the intensity of continuous environmental or developmental noise. Larger $\sigma$ produces more erratic trajectories even with strong regulation.

- $J_t$ is a **compound Poisson process** describing discontinuous jumps. Jumps occur at times determined by a Poisson process with intensity $\lambda$ (average rate of jumps per unit time). When a jump occurs, its magnitude is drawn from a distribution, here taken as $N(\mu_J, \sigma_J^2)$. This captures developmental transitions like metamorphosis or environmental regime shifts.

### Analytical Properties

The OU process with jumps admits analytical solutions for key statistical quantities, enabling efficient parameter estimation and model validation.

**Stationary Distribution**: Under $\kappa > 0$ (ensuring mean reversion), the process converges to a stationary distribution regardless of initial condition. After sufficiently long time:

$$X_\infty \sim N\left(\theta, \frac{\sigma^2}{2\kappa} + \frac{\lambda(\sigma_J^2 + \mu_J^2)}{\kappa}\right) \label{eq:ou_stationary}$$

The mean equals $\theta$ (the equilibrium), while the variance has two components: $\sigma^2/(2\kappa)$ from continuous diffusion and $\lambda(\sigma_J^2 + \mu_J^2)/\kappa$ from jumps. Notice that stronger regulation (larger $\kappa$) reduces variance, while more frequent or larger jumps (larger $\lambda$, $\mu_J$, or $\sigma_J$) increase variance.

**Autocorrelation Function**: The correlation between observations at times $s$ and $t$ decays exponentially:

$$\text{Corr}(X_s, X_t) = e^{-\kappa|t-s|} \label{eq:ou_autocorr}$$

This exponential decay is a signature of the OU process. The decay rate $\kappa$ determines how quickly past values become uninformative about future values. In biological terms, this quantifies the "memory" of the developmental system.

**Conditional Moments**: Given current state $X_s = x$, we can predict future values. The expected value at time $t > s$ is:

$$\mathbb{E}[X_t|X_s = x] = \theta + (x - \theta)e^{-\kappa(t-s)} + \lambda\mu_J(t-s) \label{eq:ou_conditional_mean}$$

This shows relaxation from initial value $x$ toward equilibrium $\theta$, plus accumulation of expected jumps. The conditional variance is:

$$\text{Var}[X_t|X_s = x] = \frac{\sigma^2}{2\kappa}(1 - e^{-2\kappa(t-s)}) + \lambda(\sigma_J^2 + \mu_J^2)(t-s) \label{eq:ou_conditional_var}$$

Initially (small $t-s$), variance is small: the current state strongly predicts the near future. As $t-s$ increases, uncertainty grows, asymptoting to the stationary variance.

### Parameter Estimation

We employ maximum likelihood estimation (see Section 5 for computational implementation details). The log-likelihood for observed data $\{x_{t_0}, x_{t_1}, \ldots, x_{t_n}\}$ is:

$$\ell(\kappa, \theta, \sigma, \lambda) = \sum_{i=1}^{n} \log p(x_{t_i} | x_{t_{i-1}}) \label{eq:ou_likelihood}$$

where the transition density can be computed using characteristic functions or numerical methods. Results from applying this estimation procedure to synthetic and real data are presented in Section 6.

## Fractional Brownian Motion

### Definition and Properties

Fractional Brownian motion (fBM) generalizes standard Brownian motion to incorporate **long-range temporal dependencies**—correlations between events separated by long time intervals. Unlike standard Brownian motion where past and future are independent given the present, fBM exhibits memory: the direction of past changes influences the direction of future changes.

Formally, fBM is a continuous Gaussian process $B^H_t$ defined by:

$$B^H_0 = 0, \quad \mathbb{E}[B^H_t] = 0 \label{eq:fbm_zero_mean}$$
$$\mathbb{E}[B^H_t B^H_s] = \frac{1}{2}(t^{2H} + s^{2H} - |t-s|^{2H}) \label{eq:fbm_covariance}$$

Here $H \in (0, 1)$ is the **Hurst parameter** controlling the correlation structure. The covariance formula shows that correlations depend on time separation $|t-s|$ in a power-law fashion, rather than the exponential decay seen in OU processes. This enables modeling of developmental systems where early events have persistent effects across ontogeny.

### Long-Range Dependence

The Hurst parameter $H$ fundamentally determines the character of temporal correlations:

- $H = 0.5$: Standard Brownian motion with **independent increments**. The past provides no information about future direction. This is the "memoryless" case appropriate for processes where fluctuations at different times are uncorrelated.

- $H > 0.5$: **Persistent motion** with positive correlations. Positive changes tend to be followed by positive changes; negative changes by negative changes. The process exhibits momentum or trending behavior. In developmental biology, this models situations where constraints or cascades cause developmental trajectories to persist in their current direction—think of developmental channeling or epigenetic inheritance.

- $H < 0.5$: **Anti-persistent motion** with negative correlations. Positive changes tend to be followed by negative changes, producing mean-reverting oscillatory behavior different from OU processes. This might model developmental systems with negative feedback operating on slow timescales.

The autocorrelation of increments decays as a power law:

$$\rho(k) \sim H(2H-1)k^{2H-2} \text{ as } k \to \infty \label{eq:fbm_autocorr_decay}$$

For $H > 0.5$, this decays slowly (e.g., as $k^{-0.4}$ when $H=0.7$), meaning correlations persist over long time lags. This "long memory" distinguishes fBM from short-memory processes like OU where correlations decay exponentially fast.

### Simulation Method

We use the Davies-Harte method for exact simulation. The covariance matrix of increments is:

$$\Gamma_{ij} = \frac{1}{2}[\Delta t_i^{2H} + \Delta t_j^{2H} - |\Delta t_i - \Delta t_j|^{2H}] \label{eq:fbm_cov_matrix}$$

Increments are generated as:
$$\Delta X \sim \mathcal{N}(0, \Gamma) \label{eq:fbm_simulation}$$

### Hurst Parameter Estimation

We estimate $H$ using the variance method. For lag $k$:
$$\mathbb{E}[(X_{t+k} - X_t)^2] = \sigma^2 k^{2H} \label{eq:fbm_variance}$$

Taking logarithms:
$$\log \mathbb{E}[(X_{t+k} - X_t)^2] = \log \sigma^2 + 2H \log k \label{eq:fbm_log_variance}$$

We estimate $H$ by regressing $\log \text{Var}(\Delta_k X)$ on $\log k$.

## Cox-Ingersoll-Ross Process

### Model Specification

The Cox-Ingersoll-Ross (CIR) process, originally developed for modeling interest rates in finance (Cox et al. 1985), provides an elegant solution to a common biological challenge: modeling mean-reverting traits that must remain non-negative (e.g., population sizes, gene expression levels, resource concentrations):

$$dX_t = \kappa(\theta - X_t)dt + \sigma\sqrt{X_t}dW_t \label{eq:cir_sde}$$

The key innovation is the **square-root diffusion term** $\sigma\sqrt{X_t}$. Unlike the OU process where noise magnitude is constant, here noise scales with $\sqrt{X_t}$:

- When $X_t$ is large, noise is substantial (in absolute terms), but small relative to $X_t$
- When $X_t$ approaches zero, noise vanishes, preventing the process from becoming negative
- This creates a "reflecting boundary" at zero without introducing artificial hard constraints

Biologically, this models phenomena where variability scales with population size or concentration—consistent with demographic stochasticity in small populations or molecular counting noise at low expression levels.

### Non-Central Chi-Square Distribution

The CIR process admits an exact analytical solution for its transition density. Under the **Feller condition** $2\kappa\theta \geq \sigma^2$ (which guarantees the process never reaches zero), the conditional distribution follows a scaled non-central chi-square:

$$\frac{2\kappa}{\sigma^2(1-e^{-\kappa\Delta t})}X_{t+\Delta t} | X_t \sim \chi'^2\left(\delta, \lambda\right) \label{eq:cir_distribution}$$

where:
- $\delta = \frac{4\kappa\theta}{\sigma^2}$ (degrees of freedom)—measures the "strength" of mean reversion relative to noise
- $\lambda = \frac{2\kappa X_t e^{-\kappa\Delta t}}{\sigma^2(1-e^{-\kappa\Delta t})}$ (non-centrality parameter)—encodes dependence on current state

This analytical tractability enables efficient maximum likelihood estimation and simulation.

### Stationary Distribution

When the Feller condition holds, the process has a Gamma stationary distribution:

$$X_\infty \sim \text{Gamma}\left(\frac{2\kappa\theta}{\sigma^2}, \frac{2\kappa}{\sigma^2}\right) \label{eq:cir_stationary}$$

The mean is $\theta$ (matching the OU equilibrium), but unlike OU, the distribution is positively skewed with support on $(0, \infty)$. Stronger mean reversion or lower noise (larger $\kappa\theta/\sigma^2$) produces distributions more concentrated around $\theta$.

### Parameter Estimation

We use moment matching:

$$\hat{\theta} = \bar{X} \label{eq:cir_theta_hat}$$
$$\hat{\kappa} = -\frac{\log(\hat{\rho}(1))}{\Delta t} \label{eq:cir_kappa_hat}$$
$$\hat{\sigma}^2 = \frac{2\hat{\kappa}\text{Var}(X)}{\hat{\theta}} \label{eq:cir_sigma_hat}$$

where $\hat{\rho}(1)$ is the lag-1 autocorrelation.

## Lévy Processes

### $\alpha$-Stable Distributions

Lévy processes provide a framework for modeling phenomena with **heavy-tailed jump distributions**—situations where rare, extreme events occur more frequently than Gaussian models predict. This is crucial for biological systems where "black swan" events (rare large-effect mutations, catastrophic environmental changes, developmental accidents) play disproportionate roles.

A random variable $X$ has a stable distribution $S_\alpha(\beta, \gamma, \delta)$ if its characteristic function is:

$$\phi(t) = \begin{cases}
\exp\left\{-\gamma^\alpha|t|^\alpha\left(1 - i\beta\text{sign}(t)\tan\frac{\pi\alpha}{2}\right) + i\delta t\right\} & \alpha \neq 1 \\
\exp\left\{-\gamma|t|\left(1 + i\beta\frac{2}{\pi}\text{sign}(t)\log|t|\right) + i\delta t\right\} & \alpha = 1
\end{cases}\label{eq:stable_cf}$$

While this formula appears complex, each parameter has clear interpretation:

- $\alpha \in (0, 2]$ is the **stability parameter** controlling tail heaviness. Smaller $\alpha$ means heavier tails and more frequent extreme events. The Gaussian distribution corresponds to $\alpha = 2$. For $\alpha < 2$, variance is infinite—a mathematical expression of the fact that extremely large values dominate moments.

- $\beta \in [-1, 1]$ is the **skewness parameter**. When $\beta = 0$, the distribution is symmetric. Positive $\beta$ produces right skew (more frequent large positive values); negative $\beta$ produces left skew.

- $\gamma > 0$ is the **scale parameter** analogous to standard deviation (though variance may not exist). It controls the typical magnitude of fluctuations.

- $\delta \in \mathbb{R}$ is the **location parameter** analogous to the mean (which exists only for $\alpha > 1$).

The characteristic function approach is necessary because stable distributions generally lack closed-form probability density functions.

### Simulation via Chambers-Mallows-Stuck

For $\alpha \neq 1$, we generate stable random variables using:

1. Generate $U \sim \text{Uniform}(-\pi/2, \pi/2)$ and $W \sim \text{Exp}(1)$
2. Compute:
$$B = \arctan\left(\beta\tan\frac{\pi\alpha}{2}\right) / \alpha \label{eq:stable_b}$$
$$S = \left(1 + \beta^2\tan^2\frac{\pi\alpha}{2}\right)^{1/(2\alpha)} \label{eq:stable_s}$$
$$X = S\frac{\sin(\alpha(U + B))}{(\cos U)^{1/\alpha}}\left(\frac{\cos(U - \alpha(U+B))}{W}\right)^{(1-\alpha)/\alpha} \label{eq:stable_simulation}$$

### Tail Behavior

For $\alpha < 2$, stable distributions have heavy tails:
$$\mathbb{P}(|X| > x) \sim C x^{-\alpha} \text{ as } x \to \infty \label{eq:stable_tails}$$

This allows modeling of extreme developmental transitions.

## Inference Framework

### Maximum Likelihood Estimation

For a discrete-time observation $\mathbf{X} = (X_0, X_{\Delta t}, \ldots, X_{n\Delta t})$, the log-likelihood is:

$$\ell(\boldsymbol{\theta}) = \sum_{i=1}^{n} \log p(X_{i\Delta t} | X_{(i-1)\Delta t}; \boldsymbol{\theta}) \label{eq:general_mle}$$

### Method of Moments

For processes with tractable moments, we match empirical and theoretical moments:

$$\hat{\boldsymbol{\theta}} = \arg\min_{\boldsymbol{\theta}} \sum_{j=1}^{k} w_j(m_j(\mathbf{X}) - m_j(\boldsymbol{\theta}))^2 \label{eq:moment_matching}$$

where $m_j$ are moment functions.

### Bayesian Inference

We can incorporate prior information:
$$p(\boldsymbol{\theta} | \mathbf{X}) \propto p(\mathbf{X} | \boldsymbol{\theta}) p(\boldsymbol{\theta}) \label{eq:bayesian_posterior}$$

Posterior sampling via MCMC provides uncertainty quantification for parameters.
