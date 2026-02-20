import numpy as np
from dataclasses import dataclass

@dataclass
class Params:
    # Temperatures
    dT: 220   # ΔT
    Tm: 510   # T_m (mean temperature)

    # Material base values
    k0: 2.67   # thermal conductivity at reference T
    rho0: 18.8 # electrical resistivity at reference T (or density if that's what ρ0 is in your notation)
    alpha0: -304.9  # Seebeck at reference T

    # Polynomial coefficients (from your model)
    A1: 4.25e-4; A2: -2.38e-6
    B1: -1.21e-3; B2: 1.0e-6
    C1: 1.03e-4; C2: 2.75e-6

def compute_J(p: Params) -> float:
    dT, Tm = p.dT, p.Tm
    k0, rho0, a0 = p.k0, p.rho0, p.alpha0
    A1, A2, B1, B2, C1, C2 = p.A1, p.A2, p.B1, p.B2, p.C1, p.C2

    # Helper pieces (follow the exact structure of Eqs. (31)–(32))
    t1 = 480*B2*k0*rho0*dT**3 + 5760*k0*rho0*dT
    t2 = 240*a0*A2*k0*dT**4 + 2880*a0*k0*dT**2
    t3 = (-480*a0*A1*rho0*dT**2
          + 240*a0*B1*rho0*dT**2
          + 240*a0*C1*rho0*dT**2
          - 240*a0*A2*rho0*Tm*dT**2
          - 240*a0*B2*rho0*Tm*dT**2
          + 240*a0*C2*rho0*Tm*dT**2
          - 2880*a0*rho0*Tm)

    # Discriminant (numerical guard in case of tiny negative due to round-off)
    disc = t1**2 - 4*t2*t3
    if disc < 0 and np.isclose(disc, 0.0, atol=1e-12*abs(t1**2)):
        disc = 0.0
    if disc < 0:
        raise ValueError(f"Discriminant became negative ({disc:.3e}). Check parameters.")

    N = t1 - np.sqrt(disc)

    D = 2 * (-480*a0*A1*rho0*dT**2
             + 240*a0*B1*rho0*dT**2
             + 240*a0*C1*rho0*dT**2
             - 240*a0*A2*rho0*Tm*dT**2
             - 240*a0*B2*rho0*Tm*dT**2
             + 240*a0*C2*rho0*Tm*dT**2
             - 2880*a0*rho0*Tm)

    if np.isclose(D, 0.0):
        raise ZeroDivisionError("Denominator D is ~0; J would be singular. Check parameters.")

    return N / D

# ---------------------------
# Example: fill your numbers here
# ---------------------------
p = Params(
    dT=180, Tm=383,       # ΔT, Tm
    k0=1.8735, rho0 =1.137, alpha0=0.0637,  # base values
    A1=1.96e-3, A2=3.25e-6,         # polynomial coeffsi Seebeck
    B1=3.88e-3, B2=3.33e-6,    # electric 
    C1=1.96e-3, C2=3.25e-6      # thermo
)


J = compute_J(p)
print("J =", J)

