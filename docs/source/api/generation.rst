Generation Module
=================

The generation module handles the creation of harmless trajectories using various API providers.

.. automodule:: AuraGen.generation
   :members:
   :undoc-members:
   :show-inheritance:

Generator Classes
-----------------

Base Generator
~~~~~~~~~~~~~~

.. autoclass:: AuraGen.generation.BaseGenerator
   :members:
   :special-members: __init__
   :show-inheritance:

OpenAI Generator
~~~~~~~~~~~~~~~~

.. autoclass:: AuraGen.generation.OpenAIGenerator
   :members:
   :special-members: __init__
   :show-inheritance:

External API Generator
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AuraGen.generation.ExternalAPIGenerator
   :members:
   :special-members: __init__
   :show-inheritance:

Local Model Generator
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AuraGen.generation.LocalGenerator
   :members:
   :special-members: __init__
   :show-inheritance:

Configuration Classes
---------------------

Generation Settings
~~~~~~~~~~~~~~~~~~~

.. autoclass:: AuraGen.generation.GenerationSettings
   :members:
   :show-inheritance:

OpenAI Config
~~~~~~~~~~~~~

.. autoclass:: AuraGen.generation.OpenAIConfig
   :members:
   :show-inheritance:

External API Config
~~~~~~~~~~~~~~~~~~~

.. autoclass:: AuraGen.generation.ExternalAPIConfig
   :members:
   :show-inheritance:

Local Config
~~~~~~~~~~~~

.. autoclass:: AuraGen.generation.LocalConfig
   :members:
   :show-inheritance:

Factory Functions
-----------------

.. autofunction:: AuraGen.generation.create_generator

.. autofunction:: AuraGen.generation.load_generation_settings

Utility Functions
-----------------

.. autofunction:: AuraGen.generation.validate_api_key

.. autofunction:: AuraGen.generation.estimate_tokens

Examples
--------

Basic Generation
~~~~~~~~~~~~~~~~

.. code-block:: python

   from AuraGen.generation import OpenAIGenerator, OpenAIConfig
   
   # Configure OpenAI generator
   config = OpenAIConfig(
       api_key_type="openai_api_key",
       model="gpt-4o",
       temperature=1.0,
       max_tokens=2048
   )
   
   generator = OpenAIGenerator(config)
   
   # Generate a single trajectory
   trajectory = generator.generate_single(
       scenario_name="email_assistant",
       constraints={"industry": "technology"}
   )
   
   print(f"Generated: {trajectory}")

Batch Generation
~~~~~~~~~~~~~~~~

.. code-block:: python

   from AuraGen.generation import create_generator, load_generation_settings
   
   # Load settings from file
   settings = load_generation_settings("config/generation.yaml")
   
   # Create appropriate generator based on settings
   generator = create_generator(settings)
   
   # Generate multiple trajectories
   trajectories = generator.generate_batch(
       scenario_names=["email_assistant", "calendar_manager"],
       batch_size=10
   )
   
   print(f"Generated {len(trajectories)} trajectories")

Custom Generator
~~~~~~~~~~~~~~~~

.. code-block:: python

   from AuraGen.generation import BaseGenerator
   from AuraGen.models import Trajectory
   
   class CustomGenerator(BaseGenerator):
       def __init__(self, custom_config):
           super().__init__()
           self.config = custom_config
       
       def _generate_single_impl(self, prompt: str) -> str:
           # Custom generation logic
           return self.custom_api_call(prompt)
       
       def custom_api_call(self, prompt: str) -> str:
           # Your custom API integration
           pass
   
   # Use custom generator
   generator = CustomGenerator(my_config)
   trajectory = generator.generate_single("test_scenario")

Error Handling
~~~~~~~~~~~~~~

.. code-block:: python

   from AuraGen.generation import OpenAIGenerator, GenerationError
   
   try:
       generator = OpenAIGenerator(config)
       trajectory = generator.generate_single("scenario")
   except GenerationError as e:
       print(f"Generation failed: {e}")
       # Handle the error appropriately
   except Exception as e:
       print(f"Unexpected error: {e}")

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AuraGen.generation import ExternalAPIGenerator
   import asyncio
   
   async def generate_concurrent():
       generator = ExternalAPIGenerator(config)
       
       # Generate multiple trajectories concurrently
       tasks = [
           generator.generate_single_async("scenario_1"),
           generator.generate_single_async("scenario_2"),
           generator.generate_single_async("scenario_3")
       ]
       
       trajectories = await asyncio.gather(*tasks)
       return trajectories
   
   # Run concurrent generation
   trajectories = asyncio.run(generate_concurrent())
