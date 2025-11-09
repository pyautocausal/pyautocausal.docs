# Difference-in-Differences Decision Flow

This guide provides a step-by-step decision flow for implementing Difference-in-Differences (DiD) analyses.

## Decision Tree

START
│
├─► 1. Nail down the causal target (ATT of what population? when?)
│
├─► 2. What does your panel look like?
│       │
│       ├─ Exactly 2 periods & 2 groups → Classic 2×2 DiD  
│       │     Estimator = ΔYtreated − ΔYcontrol.  Stop here.
│       │
│       └─ More periods / staggered entry → go to 3
│
├─► 3. Pick a comparison group assumption
│       │
│       ├─ **Never-treated units available & credible?**  
│       │        Yes → Assumption PT-GT-Nev → Estimator:  
│       │              Callaway-Sant'Anna "never" or Sun-Abraham cohort-time averages
│       │
│       └─ No → Use not-yet-treated units  
│                Assumption PT-GT-NYT → Estimator:  
│                Callaway-Sant'Anna "nyt", stacked DiD, local projections
│
├─► 4. Will you impose *full* parallel trends across **all** groups & times?  
│       │           (PT-GT-all — strong and risky)  
│       ├─ Yes → Estimator options: Extended TWFE (Wooldridge 2021),  
│       │        Borusyak-Jaravel-Spiess, de Chaisemartin-D'Haultfoeuille  
│       └─ No  → stick with comparison group choice from step 3
│
├─► 5. Are treatment effects plausibly constant across cohorts/time?  
│       │
│       ├─ Yes → You *may* report two-way fixed-effects (TWFE)  
│       │        but first decompose the weights (e.g., `bacondecomp`)  
│       │        to check for negative or perverse weighting.  
│       │
│       └─ No  → Drop TWFE; rely on block-based estimators.  
│                 (TWFE mixes already-treated controls and can flip signs.)
│
├─► 6. Covariate imbalance between treated and controls?
│       │
│       ├─ No → stay unconditional.  
│       │
│       └─ Yes → Impose **Conditional Parallel Trends** and choose:  
│                • Regression Adjustment (RA)  
│                • Inverse Probability Weighting (IPW)  
│                • Doubly-Robust (DR) combo  
│                (Sant'Anna-Zhao 2020; Callaway-Sant'Anna 2021)
│
├─► 7. Decide the weighting scheme ω up-front  
│       (unit-level ATT vs person-level ATT; weights shape the estimand)
│
├─► 8. Inference & robustness  
│       • Cluster appropriately; discuss what's random (Guide Step 4)
│       • Plot pre-trends for every comparison block.  
│       • Run sensitivity/bounds if pre-trends look shaky (Rambachan-Roth etc.).
│
└─► 9. If any assumption above fails → abandon DiD or redesign.  
        (Forward-engineer, don't reverse-engineer.)
END

## Key Principles

- **Forward-engineer your design**: Start with the causal target and work forward through assumptions.
- **Check assumptions rigorously**: Use diagnostic plots and tests to validate your approach.
- **Be transparent about limitations**: Document assumptions and potential violations.
