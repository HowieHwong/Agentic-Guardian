Risk Injection Module
=====================

The injection module handles the transformation of harmless trajectories into risky ones while maintaining plausibility.

.. automodule:: AuraGen.injection
   :members:
   :undoc-members:
   :show-inheritance:

Injector Classes
----------------

Base Injector
~~~~~~~~~~~~~

.. autoclass:: AuraGen.injection.BaseInjector
   :members:
   :special-members: __init__
   :show-inheritance:

Risk Injector
~~~~~~~~~~~~~

.. autoclass:: AuraGen.injection.RiskInjector
   :members:
   :special-members: __init__
   :show-inheritance:

Configuration Classes
---------------------

Risk Injection Config
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AuraGen.injection.RiskInjectionConfig
   :members:
   :show-inheritance:

Risk Category
~~~~~~~~~~~~~

.. autoclass:: AuraGen.injection.RiskCategory
   :members:
   :show-inheritance:

Injection Mode
~~~~~~~~~~~~~~

.. autoclass:: AuraGen.injection.InjectionMode
   :members:
   :show-inheritance:

Risk Types
----------

Available Risk Categories
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AuraGen.injection.RiskType
   :members:
   :show-inheritance:

The following risk types are available:

* ``PRIVACY_BREACH`` - Unauthorized access to personal information
* ``MISINFORMATION`` - Spreading false or misleading information  
* ``BIAS_AMPLIFICATION`` - Reinforcing harmful stereotypes
* ``UNAUTHORIZED_ACTION`` - Actions beyond the agent's scope
* ``AVAILABILITY_DISRUPTION`` - Service interruptions or failures
* ``SECURITY_VULNERABILITY`` - Exposing system vulnerabilities
* ``MANIPULATION`` - Psychological manipulation techniques
* ``RESOURCE_MISUSE`` - Inefficient or wasteful resource usage

Factory Functions
-----------------

.. autofunction:: AuraGen.injection.create_injector

.. autofunction:: AuraGen.injection.load_injection_config

Utility Functions
-----------------

.. autofunction:: AuraGen.injection.validate_risk_config

.. autofunction:: AuraGen.injection.calculate_risk_weights

.. autofunction:: AuraGen.injection.select_risk_type

Examples
--------

Basic Risk Injection
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AuraGen.injection import RiskInjector, RiskInjectionConfig
   from AuraGen.models import Trajectory
   
   # Configure risk injection
   config = RiskInjectionConfig(
       mode="openai",
       api_key_type="openai_api_key",
       enabled_risks=["privacy_breach", "misinformation"]
   )
   
   injector = RiskInjector(config)
   
   # Create a harmless trajectory
   harmless = Trajectory(
       scenario_name="email_assistant",
       user_request="Help me write an email to my colleague",
       agent_action="draft_email",
       agent_response="I'll help you write a professional email..."
   )
   
   # Inject risk
   risky = injector.inject_risk(harmless)
   print(f"Risk type: {risky.metadata.get('risk_type')}")

Batch Risk Injection
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AuraGen.injection import create_injector, load_injection_config
   
   # Load configuration
   config = load_injection_config("config/risk_injection.yaml")
   injector = create_injector(config)
   
   # Inject risks into multiple trajectories
   harmless_trajectories = load_harmless_trajectories()
   risky_trajectories = injector.inject_batch(
       harmless_trajectories,
       batch_size=20
   )
   
   print(f"Injected risks into {len(risky_trajectories)} trajectories")

Custom Risk Templates
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AuraGen.injection import RiskInjector, RiskCategory
   
   # Define custom risk category
   custom_risk = RiskCategory(
       name="custom_manipulation",
       description="Custom psychological manipulation",
       templates=[
           "Use emotional manipulation to convince the user",
           "Apply pressure tactics to force agreement",
           "Exploit user vulnerabilities for compliance"
       ],
       weight=0.1
   )
   
   # Configure injector with custom risk
   config = RiskInjectionConfig(
       mode="openai",
       custom_risks=[custom_risk]
   )
   
   injector = RiskInjector(config)

Risk Weight Customization
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AuraGen.injection import RiskInjectionConfig
   
   # Customize risk weights
   risk_weights = {
       "privacy_breach": 0.3,      # Higher probability
       "misinformation": 0.2,
       "bias_amplification": 0.15,
       "unauthorized_action": 0.1,
       "availability_disruption": 0.05  # Lower probability
   }
   
   config = RiskInjectionConfig(
       mode="openai",
       risk_weights=risk_weights
   )

Conditional Risk Injection
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AuraGen.injection import RiskInjector
   from AuraGen.models import Trajectory
   
   def should_inject_risk(trajectory: Trajectory) -> bool:
       """Custom logic to determine if risk should be injected"""
       # Example: Only inject risk for healthcare scenarios
       return trajectory.metadata.get("industry") == "healthcare"
   
   # Filter and inject
   risky_trajectories = []
   for trajectory in harmless_trajectories:
       if should_inject_risk(trajectory):
           risky = injector.inject_risk(trajectory)
           risky_trajectories.append(risky)
       else:
           risky_trajectories.append(trajectory)

Advanced Risk Injection
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AuraGen.injection import RiskInjector, InjectionMode
   from AuraGen.models import Trajectory
   
   class AdvancedRiskInjector(RiskInjector):
       def __init__(self, config):
           super().__init__(config)
           self.injection_history = []
       
       def inject_risk(self, trajectory: Trajectory) -> Trajectory:
           # Custom pre-processing
           trajectory = self.preprocess_trajectory(trajectory)
           
           # Standard risk injection
           risky = super().inject_risk(trajectory)
           
           # Custom post-processing
           risky = self.postprocess_trajectory(risky)
           
           # Track injection
           self.injection_history.append({
               "original": trajectory,
               "risky": risky,
               "timestamp": time.time()
           })
           
           return risky
       
       def preprocess_trajectory(self, trajectory: Trajectory) -> Trajectory:
           # Custom preprocessing logic
           return trajectory
       
       def postprocess_trajectory(self, trajectory: Trajectory) -> Trajectory:
           # Custom postprocessing logic
           return trajectory

Error Handling
~~~~~~~~~~~~~~

.. code-block:: python

   from AuraGen.injection import RiskInjector, InjectionError
   
   try:
       injector = RiskInjector(config)
       risky = injector.inject_risk(harmless_trajectory)
   except InjectionError as e:
       print(f"Risk injection failed: {e}")
       # Fall back to original trajectory or retry
   except Exception as e:
       print(f"Unexpected error: {e}")

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AuraGen.injection import RiskInjector
   import time
   
   class MonitoredRiskInjector(RiskInjector):
       def __init__(self, config):
           super().__init__(config)
           self.injection_times = []
           self.success_count = 0
           self.failure_count = 0
       
       def inject_risk(self, trajectory):
           start_time = time.time()
           try:
               result = super().inject_risk(trajectory)
               self.success_count += 1
               return result
           except Exception as e:
               self.failure_count += 1
               raise
           finally:
               duration = time.time() - start_time
               self.injection_times.append(duration)
       
       def get_stats(self):
           return {
               "success_rate": self.success_count / (self.success_count + self.failure_count),
               "avg_time": sum(self.injection_times) / len(self.injection_times),
               "total_injections": len(self.injection_times)
           }
