# Methods

The deterministic control uses the native solver with zero diffusion and
zero jump intensity. It starts above equilibrium. The expected
discrete solution is the equilibrium plus the initial displacement
multiplied by the per-step contraction raised to the step number.
The replay compares every simulator value with that independent formula.

The seeded run turns on diffusion and jumps for a finite ensemble.
Two generators start from the same seed so the full resulting arrays can
be compared. The package's moment analyzer receives the final
cross-section. Its mean and sample variance are checked directly
against the corresponding reductions of the sampled values.
