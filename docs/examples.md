# Example Pipelines

This section showcases real-world applications of PyAutoCausal's automated causal inference pipeline. Each example demonstrates how the pipeline automatically selects appropriate methods based on data characteristics.

## Overview

PyAutoCausal provides a pre-built pipeline (`create_panel_graph`) that:

1. **Analyzes your data structure** - Detects whether you have single or multiple treated units, treatment timing, and available control groups
2. **Selects appropriate methods** - Routes your analysis through the right causal inference techniques
3. **Generates comprehensive outputs** - Produces plots, statistical results, and executable notebooks

The pipeline uses a graph-based architecture where nodes represent analysis steps and edges show dependencies. Decision nodes (shown as diamonds) determine which path to take based on your data.

## Example 1: California Proposition 99 (Synthetic Control)

### Dataset

The California Proposition 99 dataset examines the impact of California's 1988 tobacco control program on cigarette consumption. This is a classic case study for synthetic control methods.

**Data characteristics:**
- **Single treated unit**: California (treated in 1989)
- **38 control states**: Never received the treatment
- **Time periods**: 1970-2000
- **Outcome variable**: Per capita cigarette sales

### What the Pipeline Does

When you run the pipeline on this data, it automatically detects:
- Single treated unit → Routes to synthetic control methods
- Multiple pre-treatment periods → Enables parallel trends testing
- Multiple post-treatment periods → Allows for dynamic effect estimation

The pipeline then executes:
1. **Synthetic DiD** - Modern doubly-robust synthetic control estimator
2. **Hainmueller method** - Traditional synthetic control with placebo tests
3. **Balance diagnostics** - Checks how well synthetic control matches pre-treatment California

### Pipeline Execution Graph

The graph below shows the actual execution path taken for this analysis. Green nodes were executed, while gray nodes were skipped based on data characteristics.

```{mermaid}
graph TD
    node0[df]
    node1[basic_cleaning]
    node2[panel_cleaned_data]
    node3{single_treated_unit}
    node4{multi_post_periods}
    node5{stag_treat}
    node6[did_spec]
    node7[did_spec_balance]
    node8[did_spec_balance_table]
    node9[did_spec_balance_plot]
    node10[ols_did]
    node11[save_ols_did]
    node12[event_spec]
    node13[event_spec_balance]
    node14[event_spec_balance_table]
    node15[event_spec_balance_plot]
    node16[ols_event]
    node17[event_plot]
    node18[save_event_output]
    node19[synthdid_spec]
    node20[synthdid_spec_balance]
    node21[synthdid_spec_balance_table]
    node22[synthdid_spec_balance_plot]
    node23[synthdid_fit]
    node24[synthdid_plot]
    node25[hainmueller_fit]
    node26[hainmueller_placebo_test]
    node27[hainmueller_effect_plot]
    node28[hainmueller_validity_plot]
    node29[hainmueller_output]
    node30[stag_spec]
    node31[stag_spec_balance]
    node32[stag_spec_balance_table]
    node33[stag_spec_balance_plot]
    node34[ols_stag]
    node35{has_never_treated}
    node36[cs_never_treated]
    node37[cs_never_treated_plot]
    node38[cs_never_treated_group_plot]
    node39[cs_not_yet_treated]
    node40[cs_not_yet_treated_plot]
    node41[cs_not_yet_treated_group_plot]
    node42[stag_event_plot]
    node43[save_stag_output]
    node0 --> node1
    node1 --> node2
    node2 --> node3
    node3 -->|False| node4
    node3 -->|True| node19
    node4 -->|True| node5
    node4 -->|False| node6
    node5 -->|False| node12
    node5 -->|True| node30
    node6 --> node7
    node7 --> node8
    node7 --> node9
    node7 --> node10
    node10 --> node11
    node12 --> node13
    node13 --> node14
    node13 --> node15
    node13 --> node16
    node16 --> node17
    node16 --> node18
    node19 --> node20
    node19 --> node25
    node20 --> node21
    node20 --> node22
    node20 --> node23
    node23 --> node24
    node25 --> node26
    node25 --> node27
    node26 --> node28
    node26 --> node29
    node30 --> node31
    node31 --> node32
    node31 --> node33
    node31 --> node34
    node31 --> node35
    node34 --> node42
    node34 --> node43
    node35 -->|True| node36
    node35 -->|False| node39
    node36 --> node37
    node36 --> node38
    node39 --> node40
    node39 --> node41

    %% Node styling
    classDef pendingNode fill:lightblue,stroke:#3080cf,stroke-width:2px,color:black;
    classDef runningNode fill:yellow,stroke:#3080cf,stroke-width:2px,color:black;
    classDef completedNode fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black;
    classDef failedNode fill:salmon,stroke:#3080cf,stroke-width:2px,color:black;
    classDef passedNode fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black;
    style node0 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node1 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node2 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node3 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node4 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node5 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node6 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node7 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node8 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node9 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node10 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node11 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node12 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node13 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node14 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node15 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node16 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node17 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node18 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node19 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node20 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node21 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node22 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node23 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node24 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node25 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node26 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node27 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node28 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node29 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node30 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node31 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node32 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node33 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node34 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node35 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node36 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node37 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node38 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node39 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node40 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node41 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node42 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node43 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
```

### Key Results

The pipeline outputs show:
- **Treatment effect**: Significant reduction in cigarette sales after Prop 99
- **Placebo tests**: Pass validity checks (synthetic controls for other states don't show similar effects)
- **Balance plots**: Synthetic California closely matches actual pre-treatment trends

Generated files can be found in:
- `examples/outputs/california_prop99/plots/` - Visualization outputs
- `examples/outputs/california_prop99/notebooks/pipeline_execution.ipynb` - Reproducible notebook
- `examples/outputs/california_prop99/text/` - Statistical results and summaries

## Example 2: Minimum Wage Study (Staggered DiD)

### Dataset

This example uses county-level data on minimum wage increases across U.S. counties. Different counties adopted minimum wage increases at different times, making this a staggered adoption design.

**Data characteristics:**
- **Multiple treated units**: 500+ counties
- **Staggered treatment**: Counties treated at different times (2004-2007)
- **Never-treated units**: Available as controls
- **Time periods**: 2003-2007
- **Outcome variable**: Log employment (lemp)

### What the Pipeline Does

The pipeline detects:
- Multiple treated units → Not a synthetic control case
- Multiple post-treatment periods → Enables event studies
- Staggered treatment timing → Routes to modern DiD estimators

The pipeline then executes:
1. **Callaway-Sant'Anna DiD** - Handles treatment effect heterogeneity across cohorts
2. **Event study plots** - Shows dynamic effects over time
3. **Group-specific effects** - Estimates treatment effects by cohort
4. **Balance diagnostics** - Checks covariate balance between treated and control units

### Pipeline Execution Graph

```{mermaid}
graph TD
    node0[df]
    node1[basic_cleaning]
    node2[panel_cleaned_data]
    node3{single_treated_unit}
    node4{multi_post_periods}
    node5{stag_treat}
    node6[did_spec]
    node7[did_spec_balance]
    node8[did_spec_balance_table]
    node9[did_spec_balance_plot]
    node10[ols_did]
    node11[save_ols_did]
    node12[event_spec]
    node13[event_spec_balance]
    node14[event_spec_balance_table]
    node15[event_spec_balance_plot]
    node16[ols_event]
    node17[event_plot]
    node18[save_event_output]
    node19[synthdid_spec]
    node20[synthdid_spec_balance]
    node21[synthdid_spec_balance_table]
    node22[synthdid_spec_balance_plot]
    node23[synthdid_fit]
    node24[synthdid_plot]
    node25[hainmueller_fit]
    node26[hainmueller_placebo_test]
    node27[hainmueller_effect_plot]
    node28[hainmueller_validity_plot]
    node29[hainmueller_output]
    node30[stag_spec]
    node31[stag_spec_balance]
    node32[stag_spec_balance_table]
    node33[stag_spec_balance_plot]
    node34[ols_stag]
    node35{has_never_treated}
    node36[cs_never_treated]
    node37[cs_never_treated_plot]
    node38[cs_never_treated_group_plot]
    node39[cs_not_yet_treated]
    node40[cs_not_yet_treated_plot]
    node41[cs_not_yet_treated_group_plot]
    node42[stag_event_plot]
    node43[save_stag_output]
    node0 --> node1
    node1 --> node2
    node2 --> node3
    node3 -->|False| node4
    node3 -->|True| node19
    node4 -->|True| node5
    node4 -->|False| node6
    node5 -->|False| node12
    node5 -->|True| node30
    node6 --> node7
    node7 --> node8
    node7 --> node9
    node7 --> node10
    node10 --> node11
    node12 --> node13
    node13 --> node14
    node13 --> node15
    node13 --> node16
    node16 --> node17
    node16 --> node18
    node19 --> node20
    node19 --> node25
    node20 --> node21
    node20 --> node22
    node20 --> node23
    node23 --> node24
    node25 --> node26
    node25 --> node27
    node26 --> node28
    node26 --> node29
    node30 --> node31
    node31 --> node32
    node31 --> node33
    node31 --> node34
    node31 --> node35
    node34 --> node42
    node34 --> node43
    node35 -->|True| node36
    node35 -->|False| node39
    node36 --> node37
    node36 --> node38
    node39 --> node40
    node39 --> node41

    %% Node styling
    classDef pendingNode fill:lightblue,stroke:#3080cf,stroke-width:2px,color:black;
    classDef runningNode fill:yellow,stroke:#3080cf,stroke-width:2px,color:black;
    classDef completedNode fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black;
    classDef failedNode fill:salmon,stroke:#3080cf,stroke-width:2px,color:black;
    classDef passedNode fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black;
    style node0 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node1 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node2 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node3 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node4 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node5 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node6 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node7 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node8 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node9 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node10 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node11 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node12 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node13 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node14 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node15 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node16 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node17 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node18 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node19 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node20 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node21 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node22 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node23 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node24 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node25 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node26 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node27 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node28 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node29 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node30 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node31 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node32 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node33 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node34 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node35 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node36 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node37 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node38 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node39 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node40 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node41 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node42 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node43 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
```

### Key Results

The analysis reveals:
- **Average treatment effect**: Overall impact of minimum wage on employment
- **Cohort-specific effects**: Different effects for counties treated in different years
- **Dynamic effects**: How treatment effects evolve over time since adoption
- **Pre-trend tests**: Validates parallel trends assumption

Generated files can be found in:
- `examples/outputs/minimum_wage/plots/` - Visualization outputs including event studies
- `examples/outputs/minimum_wage/notebooks/pipeline_execution.ipynb` - Reproducible notebook
- `examples/outputs/minimum_wage/text/` - Statistical results and summaries

## Understanding the Graph Visualization

### Node Types

- **Rectangle nodes** (e.g., `basic_cleaning`, `ols_did`): Action nodes that perform computations
- **Diamond nodes** (e.g., `single_treated_unit?`, `stag_treat?`): Decision nodes that route execution based on conditions

### Node States

The node colors indicate execution status:

```{mermaid}
graph LR
    pendingNode[Pending]:::pendingNode ~~~ runningNode[Running]:::runningNode ~~~ completedNode[Completed]:::completedNode ~~~ failedNode[Failed]:::failedNode ~~~ passedNode[Skipped]:::passedNode

    classDef pendingNode fill:lightblue,stroke:#3080cf,stroke-width:2px,color:black;
    classDef runningNode fill:yellow,stroke:#3080cf,stroke-width:2px,color:black;
    classDef completedNode fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black;
    classDef failedNode fill:salmon,stroke:#3080cf,stroke-width:2px,color:black;
    classDef passedNode fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black;
```

- **Light Blue**: Pending (not yet executed)
- **Yellow**: Running (currently executing)
- **Light Green**: Completed (successfully executed)
- **Salmon**: Failed (encountered an error)
- **Gray**: Skipped (not executed due to decision node conditions)

## Running Your Own Analysis

To run similar analyses on your data:

```python
from pyautocausal.pipelines.example_graph import create_panel_graph
import pandas as pd

# Load your data
data = pd.read_csv("your_data.csv")

# Ensure required columns: id_unit, t, treat, y
# Optionally include covariates: x, x_1, x_2, etc.

# Create and run the pipeline
from pathlib import Path
output_path = Path("./outputs/my_analysis")
graph = create_panel_graph(output_path)
graph.fit(df=data)
```

The pipeline will automatically:
1. Detect your data structure
2. Choose appropriate causal inference methods
3. Generate plots and statistical outputs
4. Export a reproducible Jupyter notebook

For more details, see the [Getting Started](getting-started.md) guide.

