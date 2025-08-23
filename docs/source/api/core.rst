Core Module
===========

The core module provides the fundamental classes and functions for AuraGen's operation.

.. automodule:: AuraGen.core
   :members:
   :undoc-members:
   :show-inheritance:

Core Classes
------------

.. autoclass:: AuraGen.core.AuraGenCore
   :members:
   :special-members: __init__
   :show-inheritance:

Main Functions
--------------

.. autofunction:: AuraGen.core.generate_trajectories

.. autofunction:: AuraGen.core.apply_constraints

.. autofunction:: AuraGen.core.validate_scenario

Utilities
---------

.. autofunction:: AuraGen.core.load_scenario_config

.. autofunction:: AuraGen.core.save_trajectories

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from AuraGen.core import AuraGenCore
   
   # Initialize the core engine
   core = AuraGenCore(config_path="config/generation.yaml")
   
   # Generate trajectories for a specific scenario
   trajectories = core.generate_trajectories(
       scenario_name="email_assistant",
       num_records=10
   )
   
   print(f"Generated {len(trajectories)} trajectories")

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AuraGen.core import AuraGenCore
   from AuraGen.models import GenerationSettings
   
   # Custom settings
   settings = GenerationSettings(
       batch_size=5,
       temperature=0.8,
       max_tokens=1500
   )
   
   core = AuraGenCore(settings=settings)
   
   # Generate with custom constraints
   constraints = {
       "industry": "healthcare",
       "urgency_level": "high"
   }
   
   trajectories = core.generate_trajectories(
       scenario_name="medical_assistant",
       constraints=constraints,
       num_records=20
   )
