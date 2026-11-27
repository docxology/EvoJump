# Background

A cross-section extracts the values of a trajectory ensemble at a
specified time. Its mean and variance describe the distribution at that
time. Comparing cross-sections can reveal changing dispersion even when
the mean moves smoothly. EvoJump also supports stochastic and jump
components in the trajectory solver. The earlier study describes its
applications to developmental analysis [@friedman2026evojump].

The mean-reverting component is an Ornstein–Uhlenbeck process
[@uhlenbeck1930]. The implementation advances the state by a fixed time
step in the tested fixture. Its drift term is proportional to the
difference between the equilibrium and current state.
