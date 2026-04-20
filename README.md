# SFE Thermodynamic Database Generation

## Overview

This repository contains a Python workflow for generating a
thermodynamic database of Gibbs energies and phase information for
multicomponent Fe-based alloys. Calculations are performed using
Thermo-Calc TC-Python with the TCFE12 database.

The script processes a set of nominal alloy compositions and computes
temperature-dependent equilibrium properties under controlled phase
constraints. The output dataset is intended for downstream analysis
(e.g. SFE modelling or data-driven approaches).

## Methodology

### Input data

-   Excel file: `data/input.xlsx`
-   Columns: alloy compositions (wt.%) from element C to Co
-   Each row is treated as an independent alloy

### Austenitic reference state

For each nominal composition: - An austenitic equilibrium state is
determined - The FCC_A1 composition is extracted - This composition is
used as input for subsequent calculations

### Phase-restricted calculations

Three independent systems are defined: - HCP-only (HCP_A3) - FCC-only
(FCC_A1) - BCC-only (BCC_A2)

Key assumptions: - All other phases are removed - Global minimisation is
disabled

This enforces a single-phase framework and avoids competition between
phases with identical crystal structures.

### Temperature sampling

-   Range: 0--900 °C
-   Step: 10 °C

At each temperature: - Equilibrium is computed - Dominant phase is
identified - Gibbs energy is extracted - Phase fractions are stored

## Outputs

Results are stored in `results/`:

### Gibbs energy

-   `gibbs_results.csv`
-   `gibbs_results.xlsx`

### Phase data

-   `phases_results.csv`
-   `phases_results.xlsx`

### Checkpoint

-   `checkpoints/sfe_checkpoint.pkl`

## Structure

project/
├── data/ 
├── results/ 
├── checkpoints/
├── src/ 
├── README.md
└── requirements.txt

## Dependencies

-   Python ≥ 3.8
-   pandas, numpy, tqdm
-   tc_python (Thermo-Calc, licensed)

## Limitations

-   Equilibrium only (no kinetics)
-   Phase space restricted to HCP/FCC/BCC
-   Discrete temperature grid
-   Results depend on TCFE12 database
