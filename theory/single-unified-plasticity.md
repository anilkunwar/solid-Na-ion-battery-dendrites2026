# Theory: J2 Elastoplasticity Applied to All Phases in the Na-Dendrite SSB Simulator

> **Design principle.** The simulator does **not** treat the Na metal electrode as a purely elastic solid and the solid electrolyte as a purely elastic/brittle one. Instead, a **single, unified J2 (von Mises) flow rule with Hall–Petch yield stress and KAM/GND hardening proxy is applied to every grid cell in the domain** — Na metal, SE grain interiors, and grain boundaries alike. The phase identity enters only through the local material parameters $(\mu, \nu, b, \sigma_0, \dot\gamma_0, m)$ and through the local grain size $h(\mathbf{x})$ reconstructed from the phase-field grain indicator. This keeps the mechanical model internally consistent across the moving Na/SE interface, which is essential because that interface is exactly where the eigenstrain, the stress gradient, and the plastic zone all localise.

This document describes the full coupled theory implemented in `NaSSBSolver`. It is organised as: domain → multi-phase-field → electrochemistry → transport → charge → **unified J2 elastoplasticity (the focus of this README)** → temperature dependence → numerics → diagnostics.

---

## 1. Physical domain

A 2D rectangular domain is split into two regions:

- **Left — Na metal electrode:** order parameter $\eta_0 = 1$
- **Right — solid electrolyte:** $\eta_0 = 0$, tessellated into Voronoi grains (random / columnar / packed spheres)

The Na/SE interface is the reaction front that evolves during plating. The right (counter-electrode) boundary is held at $\phi = 0$; the left boundary is held at the applied voltage $V_{\text{app}}$.

---

## 2. Multi-phase-field (MPF) description of the phases

The code tracks a **list of $N+1$ order parameters**:

$$
\eta_0 = \text{Na metal}, \qquad \eta_1,\dots,\eta_N = \text{individual SE grains}
$$

with the phase constraint

$$
\sum_{i=0}^{N} \eta_i(\mathbf{x},t) = 1 .
$$

### 2.1 Evolution equation (per phase)

$$
\frac{\partial \eta_i}{\partial t}
= -L_i\,\mu_i
\;+\; S_{\text{BV}}\,\delta_{i0}
\;-\; L_0\,\Omega\,\sigma_h^{(\text{GPa})}\cdot 10^{6}\,\delta_{i0}
$$

with the **chemical potential**

$$
\mu_i = \underbrace{2A\,\eta_i(1-\eta_i)(1-2\eta_i)}_{\text{double well}}
\;+\; \underbrace{2 B_{\text{eff}}\,\eta_i\!\sum_{j\neq i}\eta_j^{2}}_{\text{anti-overlap (grain repulsion)}}
\;-\; \underbrace{\kappa\nabla^2\eta_i}_{\text{interface energy}} .
$$

- $A = w$ — barrier height (default $10^{6}$ J/m³)
- $B_{\text{eff}} = B$ for SE–SE pairs, but $B\cdot f_{\text{Na}}$ (default $f_{\text{Na}} = 0.05$) for Na–SE pairs, so the dendrite can wet GBs
- $L_0 = L_\eta$ for Na; $L_{\text{SE}} = 10^{-3} L_\eta$ keeps SE grains nearly static so the GB network survives
- The **stress-migration term** $\Omega\,\sigma_h$ couples hydrostatic stress back into Na transport

### 2.2 Lagrange projection

After each explicit step all fields are renormalised so $\sum_i \eta_i = 1$, and deep inside the Na electrode $\eta_0$ is Dirichlet-pinned to $1$.

### 2.3 Dynamic grain-boundary indicator

The GB indicators are **recomputed each step** from the evolving fields (not from the static Voronoi tessellation):

$$
\phi_g = \sum_i \eta_i^{2}, \qquad \phi_{gb} = 1-\phi_g .
$$

Because $\eta_0$ overlaps the SE grains at the moving Na front, the Na/SE interface automatically registers as a high-$\phi_{gb}$ region. Dendrites therefore experience GB-like transport **wherever they go** — the mechanism that produces preferential GB propagation.

---

## 3. Butler–Volmer electrochemistry

Na deposition,

$$
\mathrm{Na^{+} + e^{-} \rightleftharpoons Na},
$$

gives the reaction source for $\eta_0$:

$$
S_{\text{BV}}
= L_\phi\, h'(\eta_0)\,
\Big[e^{-\alpha F\eta/RT} - c_{Na^+}\,c_{e^-}\,e^{(1-\alpha)F\eta/RT}\Big]
\big(1 + w_f\phi_{gb}\big),
$$

where

- $h'(\eta_0) = 30\eta_0^{2}(1-\eta_0)^{2}$ confines the reaction to the interface
- $\eta = -\phi$ is the local overpotential
- $c_{e^-} = \eta_0 + \tfrac12(1-\eta_0)(1-\phi_g)$ is the electron concentration
- $w_f$ boosts BV at GBs

**Nucleation term** (needed because $h'(0)=0$):

$$
S_{\text{nuc}} = 0.5\,L_\phi\,\phi_{gb}\,c_{Na^+}\,\max(0, S_{\text{BV,rate}})\,\mathbf{1}[\eta_0<0.3]\cdot(1+w_f\phi_{gb}).
$$

A numerical clip limits the per-step change to 10 % of one phase-field unit.

---

## 4. Hart effective diffusivity (Na⁺ transport)

Hart interpolation,

$$
h(\eta) = 6\eta^{5} - 15\eta^{4} + 10\eta^{3},
\qquad h(0)=0,\; h(1)=1,\; h'(0)=h'(1)=0,
$$

defines the effective diffusivity

$$
\boxed{
D_{\text{eff}} = h(\eta_0)\,D_{\text{Na}}
+ (1-h(\eta_0))\big[\phi_{g,\text{eff}}D_{\text{bulk}}
+ (1-\phi_{g,\text{eff}})D_{gb}\big]
}
$$

with

$$
\phi_{g,\text{eff}} = \phi_g\,\big(1 - 4\eta_0(1-\eta_0)\big).
$$

The factor $4\eta_0(1-\eta_0)$ peaks at the interface ($\eta_0 = 0.5$), locally suppressing $\phi_g$, so **$D_{\text{eff}}$ jumps to $D_{gb}$ at the moving front** — the mechanism of GB-assisted dendrite growth.

$D_{gb} = D_{\text{bulk}}\cdot\text{ratio}$ (500 for PEO, $10^{4}$ for Na₃SbS₄, $10^{3}$ for NASICON).

---

## 5. Nernst–Planck reaction–diffusion (Na⁺)

$$
\frac{\partial c_{Na^+}}{\partial t}
= \nabla\cdot\!\Big[D_{\text{eff}}\big(\nabla c_{Na^+} + \tfrac{F}{RT}\,c_{Na^+}\nabla\phi\big)\Big]
\;-\; c_{Na^+}\,\frac{\partial\eta_0}{\partial t}.
$$

Explicit update with auto-substepping to satisfy the CFL condition $\Delta t \le 0.2\,\Delta x^{2}/D_{\max}$. BCs: $c_{Na}=0$ inside Na metal, $c_{Na}=1$ at the right boundary.

---

## 6. Charge conservation

$$
\nabla\cdot(\sigma_{\text{eff}}\nabla\phi)
= -c_0 F\,\frac{\partial\eta_0}{\partial t},
\qquad
\sigma_{\text{eff}} = h(\eta_0)\,\sigma_{\text{Na}} + (1-h(\eta_0))\,\phi_g\,\sigma_{\text{SE}} .
$$

The stiff Poisson inversion is replaced by a **relaxed update**:

$$
\phi^{\,n+1}
= \underbrace{V_{\text{app}}\Big(1-\tfrac{x}{L_x}\Big)}_{\text{baseline}}
\;+\; \mathcal{G}_\sigma\!\Big[-\tfrac{c_0 F\,\partial_t\eta_0\,\Delta x^{2}}{\sigma_{\text{eff}}}\Big],
$$

where the perturbation is Gaussian-smoothed and clipped to ±0.5 V. Inside Na the potential is held at $V_{\text{app}}$ (equipotential metal).

---

## 7. Unified J2 elastoplasticity across all phases

### 7.1 Scope and rationale

The mechanical model has **no phase-specific constitutive switch**. The same algorithmic sequence

1. solve elastic equilibrium with eigenstrain,
2. compute von Mises overstress,
3. integrate a J2 flow rule,
4. update the KAM/GND density,

is executed identically in Na metal cells, SE grain interiors, and GB cells. What varies between cells is only

- the **material parameter set** $(\mu, \nu, b, \sigma_0, \dot\gamma_0, m)$, and
- the **local grain size** $h(\mathbf{x})$ reconstructed from $\phi_g$.

This unified treatment is what allows the **Na/SE interface itself** to be a legitimate plastic zone: the eigenstrain mismatch across it is what generates the largest stresses, and the interface must be permitted to relieve those stresses by plastic flow — just as the two bulk phases are.

### 7.2 Elastic equilibrium (plane-strain, spectral)

Isotropic 3-D moduli are converted to 2-D plane-strain moduli:

$$
C^{2D}_{11}=\tfrac{C_{11}+C_{12}+2C_{44}}{2},\quad
C^{2D}_{12}=\tfrac{C_{11}+C_{12}-2C_{44}}{2},\quad
C^{2D}_{44}=C_{44}.
$$

The FFT Green's function for displacements $(u_x,u_y)$ under the body force generated by the eigenstrain is precomputed once:

$$
G_{11}=\frac{1}{\mu k^{2}}\Big[1-\frac{\lambda+\mu}{\lambda+2\mu}\frac{k_x^{2}}{k^{2}}\Big],\quad
G_{12}=-\frac{1}{\mu k^{2}}\frac{\lambda+\mu}{\lambda+2\mu}\frac{k_x k_y}{k^{2}},\quad
G_{22}=\dots
$$

with $k^{2}\to\varepsilon$ at the DC mode. The elastic solve returns strains and stresses.

### 7.3 Eigenstrain from Na insertion

$$
\varepsilon^{*}_{xx}=\varepsilon^{*}_{yy}=\beta\,\eta_0,
\qquad
\varepsilon^{*}_{xy}=0,
$$

with misfit parameter $\beta$ (0.30 for Na metal, 0.10 for PEO, 0.20 for Na₃SbS₄, 0.15 for NASICON). This is the **volume swelling** caused by Na incorporation — the driver of the whole mechanical problem, present in every phase.

The total strain entering the elastic solve is

$$
\varepsilon^{\text{tot}}_{ij} = \varepsilon^{\text{el}}_{ij} + \varepsilon^{*}_{ij} + \varepsilon^{p}_{ij},
$$

so the accumulated plastic strain from previous steps is retained and consistently added.

### 7.4 Von Mises and hydrostatic stress

$$
\sigma_{eq} = \sqrt{\tfrac12\big[(\sigma_{xx}-\sigma_{yy})^{2}+\sigma_{yy}^{2}+\sigma_{xx}^{2}+6\sigma_{xy}^{2}\big]},
\qquad
\sigma_h = \tfrac12(\sigma_{xx}+\sigma_{yy}).
$$

$\sigma_h$ is fed back into the stress-migration term of the MPF (§2.1); $\sigma_{eq}$ is fed into the flow rule below.

### 7.5 Local grain size (Hall–Petch length scale)

The local grain size is reconstructed from the phase-field grain indicator via its spectral gradient,

$$
h(\mathbf{x}) = \operatorname{clip}\!\Big(\frac{2}{|\nabla\phi_g|+\varepsilon},\; 0,\; 300\,\mu\text{m}\Big),
$$

and floored at $h \ge 2.001\,b$ for numerical stability. This is the only place where the MPF state explicitly feeds the plasticity module — but it is applied **everywhere**, in Na and SE alike. In the Na electrode $\phi_g \approx 1$ uniformly, so $h \to h_{\max}$ and the Hall–Petch contribution is small; in SE grains $\phi_g$ varies from 1 (interior) to 0 (GB), so $h$ varies smoothly across the microstructure.

### 7.6 Hall–Petch yield stress (all phases)

$$
\sigma_y(h) = \sigma_0 + \frac{\mu b}{2\pi h(1-\nu)}\ln\!\Big(\frac{h}{b}\Big),
$$

with a floor

$$
\sigma_y \to \sigma_0 + \frac{\mu}{2\pi(1-\nu)}
\qquad\text{when } h \le 2b .
$$

The parameters $\sigma_0$, $\mu$, $\nu$, $b$ are read from the material set for the chosen phase (see §7.10).

### 7.7 J2 flow rule (identical for Na and SE)

The overstress drives a power-law plastic multiplier:

$$
\dot\gamma = \dot\gamma_0\,
\Big[\frac{\langle\sigma_{eq}-\sigma_y\rangle}{\sigma_y}\Big]^{m},
$$

and the plastic-strain increment is integrated explicitly with a per-step clip:

$$
\Delta\varepsilon^{p}_{xx}=+\dot\gamma\,\Delta t,\qquad
\Delta\varepsilon^{p}_{yy}=-\tfrac12\dot\gamma\,\Delta t,\qquad
\Delta\varepsilon^{p}_{xy}=+\tfrac12\dot\gamma\,\Delta t,
$$

$$
d_{\text{inc}} = \min\!\Big(\tfrac23\dot\gamma\,\Delta t,\; 10^{-3}\Big).
$$

This is a **fixed-direction (Radial-Return-like) J2 update**: the plastic strain rate is taken along the constant deviatoric direction implied by plane-strain proportional loading. Because the update is volumetric-neutral (trace-free) and always deviatoric-positive, it is valid for both metals and for the (idealised) pressure-insensitive J2 response assumed for the SE phases here. The accumulated plastic strains $\varepsilon^{p}_{ij}$ are clipped to $[-0.1, 0.1]$.

**Plastic strain magnitude** (used as a diagnostic and stored in the history):

$$
|\varepsilon^{p}| = \sqrt{\tfrac23\Big[(\varepsilon^{p}_{xx})^{2} + (\varepsilon^{p}_{yy})^{2} + 2(\varepsilon^{p}_{xy})^{2}\Big]}.
$$

### 7.8 KAM / GND density (all phases)

A geometrically-necessary-dislocation (GND) density proxy is computed from the gradient of the **total** shear strain, weighted by the dynamic GB indicator:

$$
\rho_{GND}
= \frac{|\nabla\varepsilon^{\text{tot}}_{xy}|}{\max|\nabla\varepsilon^{\text{tot}}_{xy}|}
\;\cdot\;\big(0.5 + 0.5\,\phi_{gb}\big),
$$

clipped to $[0,1]$. The GB weighting reflects the physical fact that strain gradients — and therefore GND storage — concentrate at interfaces; because $\phi_{gb}$ is now **dynamic**, this concentration automatically follows the moving Na front as well as the static GB network. $\rho_{GND}$ is reported for every cell, in every phase.

### 7.9 Coupling back into the phase-field

Three independent feedbacks make the mechanical model non-passive:

1. **Stress migration** — $\Omega\sigma_h$ enters $\partial_t\eta_0$ (§2.1).
2. **Eigenstrain** — $\beta\eta_0$ re-enters the elastic solve at every step (§7.3).
3. **Plastic history** — $\varepsilon^{p}_{ij}$ accumulates and re-enters the total strain (§7.3), so the mechanical state is path-dependent.

### 7.10 Material parameter sets used in the unified J2 update

| Phase | $\mu$ (GPa) | $\nu$ | $b$ (nm) | $\sigma_0$ (MPa) | $\dot\gamma_0$ (s⁻¹) | $m$ | $\beta$ |
|---|---|---|---|---|---|---|---|
| Na metal (BCC) | 3.16 | 0.45 | 0.428 | 0.5 | $10^{-3}$ | 20 | 0.30 |
| PEO / NaTFSI | 0.18 | 0.40 | 0.428 | 5 | $10^{-3}$ | 10 | 0.10 |
| Na₃SbS₄ | 12 | 0.30 | 0.428 | 50 | $10^{-3}$ | 20 | 0.20 |
| NASICON | 50 | 0.25 | 0.428 | 200 | $10^{-3}$ | 20 | 0.15 |

**Implementation note.** In the current code path, `compute_plasticity_step` is called with `self.mat_se['plasticity']` for the whole domain, so the same $(\mu,\nu,b,\sigma_0)$ of the selected SE is used everywhere. This is a deliberate simplification for the first release: it keeps the J2 update **algorithmically uniform** (as required by the design principle above) and avoids the discontinuity that a per-cell material lookup would introduce at the moving interface without a proper interface-conforming discretisation. Replacing the single call by a per-cell lookup of the phase-resolved parameter set is the natural next step and requires no change to the update algorithm — only to the array that supplies $(\mu,\nu,b,\sigma_0)$ to it.

---

## 8. Temperature dependence — Arrhenius correction

All transport coefficients scale as

$$
X(T) = X_{\text{ref}}\exp\!\Big[\frac{E_a}{R}\Big(\frac{1}{T_{\text{ref}}}-\frac{1}{T}\Big)\Big],
$$

clamped to $[10^{-6}, 10^{6}]$. At $T = T_{\text{ref}} = 300$ K the literature values are reproduced; heating accelerates, cooling suppresses. Applied to:

| Property | $E_a$ (PEO) | $E_a$ (Na₃SbS₄) | $E_a$ (NASICON) | $E_a$ (Na) |
|---|---|---|---|---|
| $D$ | 40 kJ/mol | 30 kJ/mol | 22 kJ/mol | 13 kJ/mol |
| $\sigma$ | 35 kJ/mol | 30 kJ/mol | 22 kJ/mol | 5 kJ/mol |

The BV prefactor $F/RT$ uses the operating temperature directly. Recommended operating windows and phase-transition temperatures are stored per material and trigger warnings when exceeded (Na melts at 370.98 K; PEO crystallites melt at 338.15 K).

---

## 9. Numerical methods

| Operation | Method |
|---|---|
| $\partial/\partial x$, $\partial/\partial y$, $\nabla^{2}$, $\nabla\cdot$ | FFT (spectral) |
| Time integration | Explicit Euler with per-step clipping |
| CFL on diffusion | Auto-substep: $n_{\text{sub}} = \lceil \Delta t / (0.2\,\Delta x^{2}/D_{\max})\rceil$ |
| GB smoothing | Gaussian filter, $\sigma = 2\,w_{gb}/\Delta x$ |
| MPF constraint | Lagrange renormalisation $\sum_i\eta_i = 1$ |
| Elasticity | Precomputed FFT Green's function (one-time cost) |
| Plasticity | Fixed-direction J2 radial-return with per-step clip |
| Dirichlet BCs | Hard assignment (Na side, right boundary) |

---

## 10. Diagnostics tracked

- $\eta$ in GBs vs bulk (dendrite penetration)
- $c_{Na^+}$ in GBs vs bulk (ion depletion / accumulation)
- Maximum dendrite depth (µm)
- Na volume fraction
- $\phi_{\min}$, $\phi_{\max}$ (potential range)
- Average / maximum $\sigma_{eq}$ (mechanical degradation)
- $|\varepsilon^{p}|$ (plastic strain magnitude)
- $\sigma_y$ (yield stress)
- $\rho_{GND}$ (GND density)
- Plastic work, total energy (gradient + double well)

---

## Summary

The simulation couples four transport/mechanical processes at the moving Na/SE interface:

1. **Butler–Volmer kinetics** drives Na plating ($\eta_0$ grows)
2. **Hart-effective Na⁺ diffusion** — because $D_{gb}\gg D_{\text{bulk}}$ and because the dynamic $\phi_{gb}$ captures both pre-existing GBs and the moving interface, Na⁺ is supplied preferentially along GBs
3. **Nernst–Planck drift** redistributes ions under the evolving potential
4. **Unified J2 elastoplasticity** — the same flow rule, Hall–Petch yield stress, and KAM/GND density are applied to Na metal, SE grain interiors, and GBs alike, driven by the eigenstrain $\beta\eta_0$ and coupled back through stress migration

The result is **preferential dendrite propagation along grain boundaries**, quantitatively tunable via material choice (PEO / Na₃SbS₄ / NASICON), operating temperature (Arrhenius), initial interface morphology, and applied stress / voltage — with a mechanical model that remains **constitutionally identical across every phase and across the moving interface**.
