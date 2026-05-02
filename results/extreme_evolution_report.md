# Extreme Evolution Comparison: 128 vs 256-Res

Quantifying the impact of network expansion and residual connections on physical consistency in edge cases.

### Scenario: Horizon Grazer
| Metric | PINN-128 | PINN-256-Ref | PINN-256-Stiff |
| --- | --- | --- | --- |
| Max Traj Dev (M) | 5.41e+00 | 5.25e+00 | 6.61e+00 |
| Physics Res Loss | 1.48e-04 | 3.82e-04 | 1.13e-03 |
| Relative Energy Drift | 2.45e-02 | 3.21e-02 | 2.74e-02 |
| Hamiltonian Violation | 1.77e-02 | 8.10e-02 | 1.97e-02 |

### Scenario: Speed Demon
| Metric | PINN-128 | PINN-256-Ref | PINN-256-Stiff |
| --- | --- | --- | --- |
| Max Traj Dev (M) | 1.09e+01 | 1.86e+01 | 1.35e+01 |
| Physics Res Loss | 2.16e-04 | 6.24e-04 | 4.80e-05 |
| Relative Energy Drift | 3.22e-02 | 8.89e-02 | 3.00e-02 |
| Hamiltonian Violation | 2.55e-01 | 1.39e+00 | 3.16e-01 |

### Scenario: Far Voyager
| Metric | PINN-128 | PINN-256-Ref | PINN-256-Stiff |
| --- | --- | --- | --- |
| Max Traj Dev (M) | 2.88e+01 | 1.21e+02 | 1.57e+02 |
| Physics Res Loss | 1.42e-06 | 4.51e-06 | 5.69e-07 |
| Relative Energy Drift | 6.82e-02 | 2.53e-01 | 1.77e-01 |
| Hamiltonian Violation | 2.92e-01 | 2.82e-01 | 3.01e-01 |

### Scenario: Chaos Orbit
| Metric | PINN-128 | PINN-256-Ref | PINN-256-Stiff |
| --- | --- | --- | --- |
| Max Traj Dev (M) | 4.94e+00 | 5.25e+00 | 7.94e+00 |
| Physics Res Loss | 1.78e-05 | 1.15e-04 | 3.22e-04 |
| Relative Energy Drift | 1.84e-02 | 1.14e-02 | 1.64e-02 |
| Hamiltonian Violation | 1.13e-02 | 2.84e-02 | 2.17e-03 |

