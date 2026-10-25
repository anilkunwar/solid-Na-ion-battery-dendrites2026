# Phase Field Electrochemistry FFT Solver (Allen-Cahn, Cahn-Hilliard, Mechanics and Electrochemistry) in Sodium-ion Solid State Battery

## r1-r4 use J2 Plasticity 

J2 plasticity is unsuitable for polymers because it ignores pressure-dependence, volume changes, and viscoelasticity.
However, in the context of this specific spectral phase-field code, J2 is used as a computational necessity. It provides a mathematically elegant, computationally cheap way to regularize infinite stress singularities at the dendrite tip and acts as a phenomenological proxy for local polymer creep/stress-relaxation, allowing the FFT solver to run efficiently while still capturing the primary physics of dendrite propagation along grain boundaries.


## r1: VAR1 for phase states and VAR2 for grains within electrolyte

[![continuummodelnt2d](https://img.shields.io/badge/twovar1-streamlit-turquoise)](https://sodium-dendrites-in-solid-state-battery1.streamlit.app/) (basic code: the metal electrode / electrolyte phase states are represented by VAR1 and the grains within electrolyte are represented by VAR2. VAR1 and VAR2 are not coupled and so the simulation does not produce dendrite growth. VAR1 only phase state model is used for liquid electrolyte with no grains. Causes of No Dendrites Growth:  Implementation has the decoupled-η problem you described, the GB indicator is static, the BV source is over-clipped, and the noise is sub-pixel.)

## r2: a single set of VAR for the entire grains within the two  phase states (electrode and electrolyte)

[![continuummodelnt2d](https://img.shields.io/badge/onevar2-streamlit-turquoise)](https://sodium-dendrites-in-solid-state-battery2.streamlit.app/) ( Now, all grains within electrode or electrolyte  are represented by a MPF order parameters)

## r3: a single set of VAR for the entire grains within the two  phase states (electrode and electrolyte). the MPF order parameters are represented by the name "ETA"

[![continuummodelnt2d](https://img.shields.io/badge/onevar3-streamlit-blue)](https://sodium-dendrites-in-solid-state-battery3.streamlit.app/) ( Now, all grains within electrode or electrolyte  are represented by a MPF order parameters)


