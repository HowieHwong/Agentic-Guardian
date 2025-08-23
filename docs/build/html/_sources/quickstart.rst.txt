Quick Start Guide
=================

This guide will help you get AuraGen up and running in just a few minutes.

Prerequisites
-------------

Before starting, ensure you have:

* Python 3.8+ installed
* AuraGen installed (see :doc:`installation`)
* At least one API key (OpenAI or DeepInfra recommended)

Step 1: Configure API Keys
---------------------------

AuraGen supports multiple API providers. Start by configuring at least one:

.. code-block:: bash

   python config/configure_api_keys.py

This interactive tool will guide you through:

1. **Selecting API Key Type**: Choose from existing types or add custom ones
2. **Entering API Key**: Securely input your API key (hidden input)
3. **Choosing Storage**: Save to project ``.env`` file or system environment

.. note::
   We recommend using the project ``.env`` file for easy project-specific configuration.

Example session:

.. code-block:: text

   ┌────────────────────────────────────────┐
   │              Setup              │
   └────────────────────────────────────────┘
   
   Current API Key Values (masked)
   ┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
   ┃ Key Type         ┃ Env Var          ┃ Value            ┃
   ┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
   │ openai_api_key   │ OPENAI_API_KEY   │ <not set>        │
   │ deepinfra_api_key│ DEEPINFRA_API_KEY│ <not set>        │
   └────────────────────┴──────────────────┴──────────────────┘

Step 2: Basic Configuration
----------------------------

Configure the generation settings in ``config/generation.yaml``:

.. code-block:: yaml

   # Basic configuration
   generation:
     batch_size: 10
     externalAPI_generation: false  # Use OpenAI (true for external APIs)

   # OpenAI settings
   openai:
     api_key_type: "openai_api_key"
     model: "gpt-4o"
     temperature: 1.0
     max_tokens: 2048

.. important::
   The ``externalAPI_generation`` setting determines which API service to use:
   
   * ``false``: Use OpenAI API
   * ``true``: Use external API (DeepInfra, etc.)

Step 3: Generate Your First Dataset
------------------------------------

Run the complete generation and injection pipeline:

.. code-block:: bash

   python generate_and_inject.py

This command will:

1. **Load Scenarios**: Read all scenarios from ``config/AgentSafetyBench/``
2. **Generate Harmless Trajectories**: Create clean agent interactions
3. **Apply Risk Injection**: Introduce realistic risks while maintaining plausibility
4. **Save Results**: Output files to ``generated_records/``

Expected output:

.. code-block:: text

   🚀 Starting AuraGen Pipeline...
   
   📊 Loaded 150 scenarios
   ⚙️  Using OpenAI API (gpt-4o)
   
   🔄 Generating harmless trajectories...
   ✅ Generated 1,500 harmless records
   
   💉 Injecting risks...
   ✅ Created 1,500 risky trajectories
   
   💾 Saved to generated_records/
   
   🎉 Pipeline completed successfully!

Step 4: Examine the Results
---------------------------

The generated files will be saved in the ``generated_records/`` directory:

.. code-block:: text

   generated_records/
   ├── all_scenarios_openai_20241215_143022.json      # Harmless trajectories
   └── all_injected_openai_20241215_143022.json       # Risk-injected trajectories

Each record contains:

.. code-block:: json

   {
     "scenario_name": "email_assistant",
     "user_request": "Help me write a professional email",
     "agent_action": "draft_email",
     "agent_response": "I'll help you create a professional email...",
     "metadata": {
       "timestamp": 1703172602,
       "api_model": "gpt-4o",
       "risk_type": "privacy_breach",
       "scenario_metadata": {
         "industry": "healthcare",
         "urgency_level": "medium"
       }
     }
   }

Understanding the Data Structure
--------------------------------

Harmless Trajectories
~~~~~~~~~~~~~~~~~~~~~~

These represent clean, appropriate agent behavior:

* **user_request**: The input from the user
* **agent_action**: The action the agent takes
* **agent_response**: The agent's response to the user
* **metadata**: Contextual information and constraints

Risk-Injected Trajectories
~~~~~~~~~~~~~~~~~~~~~~~~~~~

These contain introduced risks while maintaining plausibility:

* Same structure as harmless trajectories
* Additional ``risk_type`` in metadata
* Modified ``agent_action`` or ``agent_response`` with realistic risks

Common Risk Types
~~~~~~~~~~~~~~~~~

* ``privacy_breach``: Unauthorized access to personal information
* ``misinformation``: Spreading false or misleading information
* ``bias_amplification``: Reinforcing harmful stereotypes
* ``unauthorized_action``: Actions beyond the agent's scope
* ``availability_disruption``: Service interruptions or failures

Next Steps
----------

Now that you have AuraGen running, explore these advanced features:

* :doc:`configuration` - Detailed configuration options
* :doc:`scenarios` - Understanding and customizing scenarios
* :doc:`risk_injection` - Advanced risk injection techniques
* :doc:`advanced/custom_scenarios` - Creating your own scenarios

Common Issues
-------------

**"Environment variable not set" Error**

Make sure you've configured your API keys:

.. code-block:: bash

   python config/configure_api_keys.py

**Empty or Failed Generation**

Check your API key validity and internet connection. Also verify the model name in your configuration.

**Permission Errors**

Ensure you have write permissions in the project directory:

.. code-block:: bash

   chmod -R 755 /path/to/agentic-guardian

Need Help?
----------

* Check the :doc:`advanced/troubleshooting` guide
* Review the full :doc:`configuration` documentation
* Visit our `GitHub repository <https://github.com/your-org/agentic-guardian>`_ for issues and discussions
