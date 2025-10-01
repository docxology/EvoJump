# Advanced Statistical Methods

## Wavelet Analysis for Multi-Scale Temporal Patterns

Biological development operates at multiple temporal scales simultaneously—from hourly gene expression oscillations to weekly morphological changes. Traditional Fourier analysis assumes stationarity, limiting its utility for developmental data where periodicities change over ontogeny. **Wavelet analysis** provides time-localized frequency decomposition, revealing when specific periodicities occur.

### Continuous Wavelet Transform

The CWT decomposes trajectories $x(t)$ using scaled and translated mother wavelets $\psi$:

$$W(a, b) = \frac{1}{\sqrt{a}}\int_{-\infty}^{\infty} x(t)\psi^*\left(\frac{t-b}{a}\right)dt \label{eq:cwt}$$

Parameters:
- **Mother wavelet** ($\psi$): Localized oscillatory template (Morlet for time-frequency localization, Mexican hat for transients)
- **Scale** ($a > 0$): Controls stretch—large $a$ for slow variations (low frequencies), small $a$ for rapid changes (high frequencies)
- **Translation** ($b$): Slides wavelet along time axis to detect when frequencies occur
- **Complex conjugation** ($*$): For complex wavelets

Result $W(a,b)$ shows which frequencies occur at which times, with $1/\sqrt{a}$ normalization preserving energy across scales.

### Power Spectrum and Applications

**Power spectrum** identifies dominant periodicities:
$$P(a, b) = |W(a, b)|^2$$

**Time-averaged power** reveals characteristic scales:
$$\bar{P}(a) = \frac{1}{T}\int_0^T |W(a, b)|^2 db$$

**Applications**:
- Developmental oscillations in gene expression data
- Critical periods and metamorphic transitions
- Multi-scale processes across temporal hierarchies

Complements stochastic process models by revealing time-localized patterns.

**Implementation** uses Morlet wavelet for optimal time-frequency localization:
$$\psi(t) = \pi^{-1/4}e^{i\omega_0 t}e^{-t^2/2}$$

Logarithmic scale selection: $a_j = a_0 2^{j/n_{\text{voices}}}$.

## Copula Methods for Trait Dependencies

Developmental traits exhibit complex dependencies through pleiotropy, developmental integration, and functional constraints. Traditional correlation analysis assumes linearity and multivariate normality, inadequate for nonlinear dependencies, asymmetries, and tail dependence.

**Copula methods** model complex multivariate dependencies by separating marginal distributions from dependence structure, enabling non-Gaussian dependencies with arbitrary marginals.

### Copula Theory

**Sklar's theorem** (1959) decomposes multivariate distributions:

$$F(x_1, \ldots, x_d) = C(F_1(x_1), \ldots, F_d(x_d)) \label{eq:sklar_theorem}$$

where $C: [0,1]^d \to [0,1]$ is the copula capturing dependence structure independent of marginals $F_i$. Operates on uniform marginals $U_i = F_i(X_i) \in [0,1]$.

Different copula families model distinct dependence patterns:

### Copula Families

**Gaussian Copula** (symmetric, no tail dependence):
$$C(u_1, u_2) = \Phi_\rho(\Phi^{-1}(u_1), \Phi^{-1}(u_2))$$
Extends correlation to non-normal marginals but lacks tail dependence.

**Clayton Copula** (lower tail dependence):
$$C(u_1, u_2) = \max\{(u_1^{-\theta} + u_2^{-\theta} - 1)^{-1/\theta}, 0\}$$
Strong lower tail dependence—small values co-occur. Models compensatory growth and pleiotropic mutation effects.

**Frank Copula** (symmetric, moderate tail dependence):
$$C(u_1, u_2) = -\frac{1}{\theta}\log\left(1 + \frac{(e^{-\theta u_1} - 1)(e^{-\theta u_2} - 1)}{e^{-\theta} - 1}\right)$$
Intermediate flexibility with weak tail dependence.

### Dependence Measures

**Kendall's $\tau$**:
$$\tau = \mathbb{P}[(X_1 - X_2)(Y_1 - Y_2) > 0] - \mathbb{P}[(X_1 - X_2)(Y_1 - Y_2) < 0] \label{eq:kendall_tau}$$

**Tail Dependence Coefficients**:

Upper: $\lambda_U = \lim_{u \to 1^-} \mathbb{P}(U_2 > u | U_1 > u) \label{eq:upper_tail_dep}$

Lower: $\lambda_L = \lim_{u \to 0^+} \mathbb{P}(U_2 \leq u | U_1 \leq u) \label{eq:lower_tail_dep}$

### Estimation

1. Transform data to uniform margins via empirical CDF
2. Fit copula by maximum likelihood or method of moments
3. Validate using goodness-of-fit tests

### Applications

- Model complex trait correlations beyond linear dependence
- Identify traits that co-vary in extreme phenotypes
- Characterize pleiotropy and genetic covariance structures

## Extreme Value Theory

Understanding extreme phenotypes is crucial for evolutionary biology. Rare, extreme individuals may experience strong selection, reveal hidden genetic variation, or indicate developmental constraints. Yet traditional statistical methods focus on central tendency and typical variation, treating extremes as outliers to be excluded rather than phenomena to be modeled.

**Extreme value theory (EVT)** provides rigorous statistical methods for analyzing tail behavior and rare events. Originally developed for engineering (flood risk, structural failure) and finance (market crashes), EVT has natural applications to biology: maximum body size achievable under constraints, probability of developmental catastrophes, risk of population extinction.

### Peaks-Over-Threshold Method

The **POT method** models values exceeding a high threshold $u$. The fundamental result (Pickands 1975) is that threshold exceedances, under mild conditions, follow a **Generalized Pareto Distribution (GPD)**:

$$F(x) = 1 - \left(1 + \xi\frac{x-u}{\sigma}\right)^{-1/\xi}_+$$

where the notation $(\cdot)_+$ means $\max(\cdot, 0)$ and:

- $\xi$ is the **shape parameter** (tail index) determining tail behavior. This parameter has profound biological interpretation:
  - $\xi > 0$: **Heavy-tailed** (Pareto-type). Extreme events far beyond the threshold occur regularly. No finite upper bound exists. This might indicate weak selection against extreme phenotypes or high mutational variance.
  - $\xi = 0$: **Exponential tail** (light but unbounded). Extremes decay exponentially. This is the boundary between bounded and unbounded distributions.
  - $\xi < 0$: **Bounded tail** (short-tailed). A finite upper bound exists at $u - \sigma/\xi$. This indicates strong developmental constraints or stabilizing selection imposing a phenotypic ceiling.

- $\sigma > 0$ is the **scale parameter** controlling the typical magnitude of exceedances, analogous to standard deviation.

### Return Levels

The $m$-observation return level satisfies:

$$\mathbb{P}(X > x_m) = \frac{1}{m} \label{eq:return_level_def}$$

For GPD with $n_u$ exceedances in $n$ observations:

$$x_m = u + \frac{\sigma}{\xi}\left[\left(m\frac{n}{n_u}\right)^\xi - 1\right] \label{eq:return_level_gpd}$$

### Block Maxima Method

Model block maxima using Generalized Extreme Value (GEV) distribution:

$$F(x) = \exp\left\{-\left(1 + \xi\frac{x-\mu}{\sigma}\right)^{-1/\xi}_+\right\} \label{eq:gev}$$

Parameters:
- $\mu \in \mathbb{R}$: location
- $\sigma > 0$: scale
- $\xi \in \mathbb{R}$: shape

### Hill Estimator

For heavy-tailed distributions, the tail index $\alpha$ is estimated by:

$$\hat{\alpha} = \left[\frac{1}{k}\sum_{i=1}^{k} \log X_{(i)} - \log X_{(k+1)}\right]^{-1} \label{eq:hill_estimator}$$

where $X_{(1)} \geq X_{(2)} \geq \ldots$ are order statistics.

### Applications

- Predict extreme developmental outcomes
- Quantify risk of pathological phenotypes
- Identify evolutionary constraints from tail behavior

## Regime Switching Detection

### Hidden Markov Models

Assume the developmental process follows regime-dependent dynamics:

$$X_t | S_t = k \sim f_k(x_t | \theta_k) \label{eq:hmm_observation}$$

where $S_t \in \{1, \ldots, K\}$ is the unobserved regime state following a Markov chain:

$$\mathbb{P}(S_t = j | S_{t-1} = i) = p_{ij} \label{eq:hmm_transition}$$

### K-Means Clustering Approach

We use sliding windows to extract features:

$$\mathbf{z}_t = [\mu_t, \sigma_t, r_t, IQR_t] \label{eq:window_features}$$

where each feature is computed over window $[t-w, t]$:
- $\mu_t$: mean
- $\sigma_t$: standard deviation
- $r_t$: range
- $IQR_t$: interquartile range

Cluster feature vectors to identify regimes.

### Transition Probability Matrix

Estimate transition probabilities:

$$\hat{p}_{ij} = \frac{\text{transitions from } i \text{ to } j}{\text{times in regime } i} \label{eq:transition_matrix}$$

### Regime Characterization

For each regime $k$:
- Mean and variance of trait values
- Duration distribution
- Proportion of total time
- Associated environmental covariates

### Applications

- Identify developmental phases
- Detect environmental regime shifts
- Characterize developmental plasticity
- Model punctuated equilibrium

## Information-Theoretic Methods

### Shannon Entropy

Quantify uncertainty in phenotypic distributions:

$$H(X) = -\int f(x) \log f(x) dx \label{eq:shannon_entropy}$$

For discrete distributions:
$$H(X) = -\sum_{i} p_i \log p_i \label{eq:discrete_entropy}$$

### Mutual Information

Measure dependence between traits:

$$I(X; Y) = \int\int f(x,y) \log\frac{f(x,y)}{f(x)f(y)} dx dy \label{eq:mutual_info}$$

### Transfer Entropy

Quantify directed information flow:

$$TE_{Y \to X} = H(X_{t+1} | X_t^{(k)}) - H(X_{t+1} | X_t^{(k)}, Y_t^{(l)}) \label{eq:transfer_entropy}$$

where $X_t^{(k)} = (X_t, X_{t-1}, \ldots, X_{t-k+1})$ is the history.

### Applications

- Quantify developmental constraints
- Identify causal relationships between traits
- Measure phenotypic integration

## Robust Statistical Methods

### M-Estimators

Robust location estimates minimize:

$$\sum_{i=1}^{n} \rho\left(\frac{x_i - \mu}{\sigma}\right) \label{eq:m_estimator}$$

**Huber M-estimator**: $\rho(u) = \begin{cases} u^2/2 & |u| \leq k \\ k|u| - k^2/2 & |u| > k \end{cases} \label{eq:huber_rho}$

**Tukey Biweight**: $\rho(u) = \begin{cases} (k^2/6)[1 - (1 - (u/k)^2)^3] & |u| \leq k \\ k^2/6 & |u| > k \end{cases} \label{eq:tukey_rho}$

### Robust Scale Estimation

**MAD (Median Absolute Deviation)**:
$$\text{MAD} = \text{median}(|X_i - \text{median}(X)|) \label{eq:mad}$$

**$Q_n$ Estimator**:
$$Q_n = c \cdot \{|X_i - X_j|; i < j\}_{(k)} \label{eq:qn_estimator}$$

where $k = \binom{h}{2}$, $h = \lfloor n/2 \rfloor + 1$, and $c \approx 2.2219$.

### Applications

- Handle outliers without manual removal
- Robust parameter estimation in presence of contamination
- Appropriate for biological data with measurement error
