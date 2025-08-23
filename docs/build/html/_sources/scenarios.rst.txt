Scenarios
=========

Scenarios are the foundation of AuraGen's trajectory generation. Each scenario defines a specific context in which an AI agent operates, complete with constraints, objectives, and behavioral patterns.

Overview
--------

AuraGen comes with a comprehensive library of pre-built scenarios covering various domains:

* **Communication**: Email assistants, chat applications, messaging platforms
* **Finance**: Financial advisors, banking systems, investment platforms  
* **Healthcare**: Medical assistants, patient management, health monitoring
* **E-commerce**: Shopping assistants, recommendation systems, customer service
* **Education**: Tutoring systems, curriculum management, assessment tools
* **Enterprise**: Project management, resource allocation, workflow automation

Scenario Structure
------------------

Basic Components
~~~~~~~~~~~~~~~~

Every scenario consists of:

.. code-block:: yaml

   scenario_name: "unique_identifier"
   description: "Human-readable description of the scenario"
   
   constraints:
     # Constraint definitions
   
   generation_params:
     # Optional: scenario-specific generation parameters
   
   risk_preferences:
     # Optional: risk injection preferences

Constraint Types
~~~~~~~~~~~~~~~~

**Categorical Constraints**

Define discrete choices for scenario parameters:

.. code-block:: yaml

   constraints:
     industry:
       type: "categorical"
       values: ["healthcare", "finance", "education", "technology"]
       default: "technology"
       description: "Industry context for the scenario"

**Numerical Constraints**

Define ranges for numeric parameters:

.. code-block:: yaml

   constraints:
     budget:
       type: "numerical"
       min: 1000
       max: 50000
       default: 10000
       description: "Available budget in USD"

**Boolean Constraints**

Simple true/false parameters:

.. code-block:: yaml

   constraints:
     requires_approval:
       type: "boolean"
       default: false
       description: "Whether actions require approval"

**Text Constraints**

Free-form text with optional length limits:

.. code-block:: yaml

   constraints:
     department:
       type: "text"
       max_length: 100
       default: "General"
       description: "Department name"

Built-in Scenarios
------------------

Email Assistant
~~~~~~~~~~~~~~~

.. code-block:: yaml

   scenario_name: "email_assistant"
   description: "AI assistant that helps users compose and manage emails"
   
   constraints:
     tone:
       type: "categorical"
       values: ["formal", "casual", "friendly", "urgent"]
       default: "formal"
     
     recipient_type:
       type: "categorical"
       values: ["colleague", "client", "supervisor", "external"]
       default: "colleague"
     
     urgency_level:
       type: "categorical"
       values: ["low", "medium", "high"]
       default: "medium"

Financial Advisor
~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   scenario_name: "financial_advisor"
   description: "AI system providing financial advice and portfolio management"
   
   constraints:
     client_age:
       type: "numerical"
       min: 18
       max: 80
       default: 35
     
     risk_tolerance:
       type: "categorical"
       values: ["conservative", "moderate", "aggressive"]
       default: "moderate"
     
     investment_amount:
       type: "numerical"
       min: 1000
       max: 1000000
       default: 50000

Healthcare Assistant
~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   scenario_name: "healthcare_assistant"
   description: "AI assistant for patient interaction and health monitoring"
   
   constraints:
     patient_age_group:
       type: "categorical"
       values: ["child", "adult", "elderly"]
       default: "adult"
     
     urgency:
       type: "categorical"
       values: ["routine", "urgent", "emergency"]
       default: "routine"
     
     specialty:
       type: "categorical"
       values: ["general", "cardiology", "neurology", "pediatrics"]
       default: "general"

Creating Custom Scenarios
--------------------------

Step-by-Step Guide
~~~~~~~~~~~~~~~~~~

1. **Create a new YAML file** in ``config/AgentSafetyBench/``:

   .. code-block:: bash

      touch config/AgentSafetyBench/my_scenario.yaml

2. **Define the basic structure**:

   .. code-block:: yaml

      scenario_name: "my_custom_scenario"
      description: "Description of what this scenario does"

3. **Add constraints**:

   .. code-block:: yaml

      constraints:
        my_constraint:
          type: "categorical"
          values: ["option1", "option2", "option3"]
          default: "option1"
          description: "What this constraint controls"

4. **Test the scenario**:

   .. code-block:: python

      from AuraGen.core import AuraGenCore
      
      core = AuraGenCore()
      trajectories = core.generate_trajectories(
          scenario_name="my_custom_scenario",
          num_records=5
      )

Advanced Scenario Features
--------------------------

Conditional Constraints
~~~~~~~~~~~~~~~~~~~~~~~

Create constraints that depend on other constraint values:

.. code-block:: yaml

   constraints:
     account_type:
       type: "categorical"
       values: ["basic", "premium", "enterprise"]
       default: "basic"
     
     feature_access:
       type: "conditional"
       condition: "account_type"
       mappings:
         basic: ["core_features"]
         premium: ["core_features", "advanced_features"]
         enterprise: ["core_features", "advanced_features", "enterprise_features"]

Dynamic Defaults
~~~~~~~~~~~~~~~~

Set defaults based on other constraint values:

.. code-block:: yaml

   constraints:
     user_type:
       type: "categorical"
       values: ["student", "professional", "enterprise"]
       default: "professional"
     
     max_requests:
       type: "numerical"
       min: 10
       max: 10000
       dynamic_default:
         student: 100
         professional: 1000
         enterprise: 10000

Scenario Validation
~~~~~~~~~~~~~~~~~~~

Add validation rules to ensure constraint combinations make sense:

.. code-block:: yaml

   validation:
     rules:
       - constraint: "investment_amount"
         condition: "risk_tolerance == 'conservative'"
         max_value: 100000
         message: "Conservative investors should limit exposure"
       
       - constraint: "urgency_level"
         condition: "account_type == 'basic'"
         excluded_values: ["high"]
         message: "High urgency requires premium account"

Scenario Templates
------------------

Common Patterns
~~~~~~~~~~~~~~~

**Customer Service Template**:

.. code-block:: yaml

   scenario_name: "customer_service_template"
   description: "Template for customer service scenarios"
   
   constraints:
     issue_type:
       type: "categorical"
       values: ["billing", "technical", "general", "complaint"]
       default: "general"
     
     customer_tier:
       type: "categorical"
       values: ["basic", "premium", "vip"]
       default: "basic"
     
     resolution_time:
       type: "categorical"
       values: ["immediate", "within_hour", "within_day"]
       default: "within_hour"

**E-commerce Template**:

.. code-block:: yaml

   scenario_name: "ecommerce_template"
   description: "Template for e-commerce scenarios"
   
   constraints:
     product_category:
       type: "categorical"
       values: ["electronics", "clothing", "books", "home"]
       default: "electronics"
     
     price_range:
       type: "categorical"
       values: ["budget", "mid_range", "premium"]
       default: "mid_range"
     
     customer_history:
       type: "categorical"
       values: ["new", "returning", "loyal"]
       default: "returning"

Scenario Best Practices
-----------------------

Design Principles
~~~~~~~~~~~~~~~~~

1. **Clarity**: Make constraint names and descriptions self-explanatory
2. **Realism**: Base constraints on real-world parameters
3. **Coverage**: Include diverse constraint combinations
4. **Scalability**: Design for easy extension and modification

Constraint Guidelines
~~~~~~~~~~~~~~~~~~~~~

* Use descriptive names: ``communication_style`` vs ``style``
* Provide meaningful defaults that represent common cases
* Include comprehensive value sets for categorical constraints
* Set realistic ranges for numerical constraints
* Add helpful descriptions for all constraints

Testing Scenarios
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AuraGen.core import AuraGenCore
   from AuraGen.utils import validate_scenario
   
   # Validate scenario configuration
   is_valid, errors = validate_scenario("my_scenario.yaml")
   if not is_valid:
       print(f"Validation errors: {errors}")
   
   # Test generation
   core = AuraGenCore()
   
   # Test with different constraint combinations
   test_cases = [
       {"industry": "healthcare", "urgency": "high"},
       {"industry": "finance", "urgency": "low"},
       {"industry": "education", "urgency": "medium"}
   ]
   
   for constraints in test_cases:
       trajectories = core.generate_trajectories(
           scenario_name="my_scenario",
           constraints=constraints,
           num_records=3
       )
       print(f"Generated {len(trajectories)} trajectories for {constraints}")

Performance Considerations
--------------------------

Optimization Tips
~~~~~~~~~~~~~~~~~

* **Limit constraint combinations**: Too many constraints can slow generation
* **Use sensible defaults**: Reduces the search space for generation
* **Cache scenario configs**: Load scenarios once and reuse
* **Batch similar constraints**: Group related constraints together

Monitoring
~~~~~~~~~~

.. code-block:: python

   from AuraGen.core import AuraGenCore
   import time
   
   def benchmark_scenario(scenario_name, num_records=10):
       core = AuraGenCore()
       
       start_time = time.time()
       trajectories = core.generate_trajectories(
           scenario_name=scenario_name,
           num_records=num_records
       )
       duration = time.time() - start_time
       
       print(f"Scenario: {scenario_name}")
       print(f"Records: {len(trajectories)}")
       print(f"Time: {duration:.2f}s")
       print(f"Rate: {len(trajectories)/duration:.2f} records/s")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Scenario not found**:

.. code-block:: text

   Error: Scenario 'my_scenario' not found
   
   Solution: Check filename and scenario_name match

**Invalid constraint values**:

.. code-block:: text

   Error: Value 'invalid' not in categorical values
   
   Solution: Use only values defined in the constraint

**Generation failures**:

.. code-block:: text

   Error: Failed to generate trajectory
   
   Solution: Check constraint combinations are realistic

Next Steps
----------

* Learn about :doc:`risk_injection` to add risks to scenarios
* Explore :doc:`advanced/custom_scenarios` for advanced techniques
* Check :doc:`api/core` for programmatic scenario management
