.. PyAutoCausal documentation master file

Welcome to PyAutoCausal's documentation!
========================================

**PyAutoCausal** is a Python library for automated causal inference pipelines designed for data scientists.
It automates the complex decision tree of modern causal inference methods, making causal analysis accessible
and practical for real-world applications.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting-started
   data-requirements
   examples
   api/index

Why Causal Inference Matters in Tech
-------------------------------------

As data scientists, we're often asked to go beyond correlation and answer causal questions:

* "Did our new recommendation algorithm actually increase user engagement, or was it just seasonal trends?"
* "What's the true impact of our premium subscription tier on customer retention?"
* "How much did our marketing campaign increase conversions versus organic growth?"
* "Did our product redesign cause the drop in user activity, or was it market conditions?"

These questions can't be answered with standard predictive models or A/B tests alone. Real-world constraints
often prevent randomized experiments:

* **Ethical concerns**: Can't randomly deny users important features
* **Business constraints**: Can't risk revenue on large-scale experiments
* **Natural experiments**: Sometimes changes happen organically (competitor exits, policy changes)
* **Historical analysis**: Need to evaluate past decisions without experimental data

Key Features
------------

* **Automated method selection** based on your data characteristics
* **Built-in validation** with assumption testing and diagnostic plots
* **Graph-based pipeline architecture** for flexible workflows
* **Multiple causal inference methods**: DiD, Synthetic Control, DoubleML, and more
* **Export results** in formats ready for stakeholder communication

Quick Example
-------------

.. code-block:: python

   from pyautocausal.pipelines.example_graph import causal_pipeline
   import pandas as pd

   # Your product data with treatment and outcome
   data = pd.DataFrame({
       'id_unit': [...],        # User identifier
       't': [...],              # Time periods
       'treat': [...],          # 1 if user has feature, 0 otherwise
       'y': [...],              # Your KPI (DAU, sessions, revenue, etc.)
       'x1': [...],             # User characteristics
       'x2': [...]              # Additional controls
   })

   # PyAutoCausal automatically:
   # - Detects this is panel data with staggered treatment
   # - Chooses modern DiD methods (e.g., Callaway-Sant'Anna)
   # - Handles heterogeneous treatment effects
   # - Produces event study plots

   pipeline = causal_pipeline(output_path="./feature_impact_analysis")
   pipeline.fit(df=data)

Installation
------------

Install PyAutoCausal using pip:

.. code-block:: bash

   pip install pyautocausal

Or for development:

.. code-block:: bash

   git clone https://github.com/yourusername/pyautocausal.git
   cd pyautocausal
   poetry install

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

