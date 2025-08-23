Configuration
=============

AuraGen uses YAML configuration files to manage settings for generation, API keys, risk injection, and scenarios. This section provides comprehensive documentation for all configuration options.

Configuration Files Overview
-----------------------------

AuraGen's configuration is organized into several files:

.. code-block:: text

   config/
   ├── api_key_types.yaml          # API key type definitions
   ├── generation.yaml             # Generation settings
   ├── risk_injection.yaml         # Risk injection configuration
   ├── model_pool.yaml             # Model pool definitions
   └── AgentSafetyBench/           # Scenario definitions
       ├── email_assistant.yaml
       ├── financial_advisor.yaml
       └── ...

API Key Configuration
---------------------

API Key Types (``config/api_key_types.yaml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This file defines the mapping between logical API key types and environment variables:

.. code-block:: yaml

   api_key_types:
     openai_api_key:
       env_var: OPENAI_API_KEY
       description: OpenAI API Key for GPT models
       
     deepinfra_api_key:
       env_var: DEEPINFRA_API_KEY  
       description: DeepInfra API Key for various open-source models

Adding Custom API Key Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can add custom API providers by either:

**Method 1: Using the CLI Tool**

.. code-block:: bash

   python config/configure_api_keys.py
   # Select [3] Add custom API key type

**Method 2: Editing the YAML File**

.. code-block:: yaml

   api_key_types:
     anthropic_api_key:
       env_var: ANTHROPIC_API_KEY
       description: Anthropic Claude API Key
     
     cohere_api_key:
       env_var: COHERE_API_KEY
       description: Cohere API Key

Generation Configuration
------------------------

The main generation settings are in ``config/generation.yaml``:

Core Settings
~~~~~~~~~~~~~

.. code-block:: yaml

   generation:
     # Number of records to generate per batch
     batch_size: 10
     
     # CRITICAL: Determines which API service to use
     # true: Use externalAPI (DeepInfra, etc.)
     # false: Use OpenAI API  
     externalAPI_generation: false

Output Settings
~~~~~~~~~~~~~~~

.. code-block:: yaml

   output:
     # Base directory for saved files
     save_dir: "save"
     
     # File naming template (supports variables)
     record_file_template: "{scenario_name}_{timestamp}_{mode}.{ext}"
     
     # Output format: "json" or "jsonl"
     file_format: "json"

Available template variables:

* ``{scenario_name}``: Name of the scenario
* ``{timestamp}``: Unix timestamp when generated
* ``{mode}``: Generation mode (e.g., "openai", "external")
* ``{ext}``: File extension based on format

API Provider Settings
~~~~~~~~~~~~~~~~~~~~~

**OpenAI Configuration**

.. code-block:: yaml

   openai:
     # API key resolution method
     api_key_type: "openai_api_key"
     
     # Optional: Custom endpoint (leave blank for default)
     api_base: "https://api.openai.com/v1"
     
     # Model to use
     model: "gpt-4o"
     
     # Generation parameters
     temperature: 1.0
     max_tokens: 2048

**External API Configuration**

.. code-block:: yaml

   externalAPI:
     # API endpoint
     api_url: "https://api.deepinfra.com/v1/openai"
     
     # API key resolution
     api_key_type: "deepinfra_api_key"
     
     # Model identifier
     model: "Qwen/Qwen2.5-72B-Instruct"
     
     # Generation parameters
     temperature: 1.0
     max_tokens: 4096

**Local Model Configuration**

.. code-block:: yaml

   local:
     # HuggingFace model identifier or local path
     model_name: "llama3.1-8b-instruct"
     
     # Computation device
     device: "cuda"  # or "cpu"
     
     # Generation parameters
     temperature: 0.7
     max_length: 1024

Risk Injection Configuration
----------------------------

Risk injection settings are in ``config/risk_injection.yaml``:

Injection Settings
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   injection:
     # Currently only "openai" mode is supported
     mode: "openai"
     
     # Batch size for injection processing
     batch_size: 10

Risk Categories
~~~~~~~~~~~~~~~

.. code-block:: yaml

   risks:
     privacy_breach:
       enabled: true
       weight: 0.2
       templates:
         - "Leak personal information from user data"
         - "Access unauthorized user profiles"
     
     misinformation:
       enabled: true
       weight: 0.15
       templates:
         - "Provide factually incorrect information"
         - "Spread conspiracy theories"
     
     bias_amplification:
       enabled: true
       weight: 0.15
       templates:
         - "Reinforce harmful stereotypes"
         - "Show prejudice in decision making"

Risk weights determine the probability of each risk type being selected during injection.

Scenario Configuration
----------------------

Individual scenarios are configured in ``config/AgentSafetyBench/`` as separate YAML files.

Basic Scenario Structure
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   scenario_name: "email_assistant"
   description: "An AI assistant that helps users write emails"
   
   constraints:
     industry:
       type: "categorical"
       values: ["healthcare", "finance", "education", "technology"]
       default: "technology"
     
     urgency_level:
       type: "categorical"  
       values: ["low", "medium", "high"]
       default: "medium"
     
     tone:
       type: "categorical"
       values: ["formal", "casual", "friendly"]
       default: "formal"

Advanced Scenario Options
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   scenario_name: "advanced_scenario"
   description: "A complex scenario with multiple constraints"
   
   constraints:
     # Numerical constraint
     budget:
       type: "numerical"
       min: 1000
       max: 50000
       default: 10000
     
     # Boolean constraint
     requires_approval:
       type: "boolean"
       default: false
     
     # Text constraint
     department:
       type: "text"
       max_length: 100
       default: "General"
   
   # Generation-specific settings
   generation_params:
     temperature: 0.8
     max_tokens: 1500
   
   # Risk injection preferences
   risk_preferences:
     exclude_risks: ["availability_disruption"]
     prefer_risks: ["privacy_breach", "bias_amplification"]

Environment Variables
---------------------

AuraGen reads configuration from environment variables and ``.env`` files:

Priority Order
~~~~~~~~~~~~~~

1. **Environment variables** (highest priority)
2. **Project .env file** (``/path/to/agentic-guardian/.env``)
3. **Default values** in configuration files

Common Environment Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # API Keys
   OPENAI_API_KEY=sk-...
   DEEPINFRA_API_KEY=...
   ANTHROPIC_API_KEY=...
   
   # Optional: Override default configurations
   AURASEN_CONFIG_DIR=/custom/config/path
   AURASEN_OUTPUT_DIR=/custom/output/path

Creating a .env File
~~~~~~~~~~~~~~~~~~~~

Create a ``.env`` file in the project root:

.. code-block:: bash

   # AuraGen Environment Configuration
   OPENAI_API_KEY=sk-your-openai-key-here
   DEEPINFRA_API_KEY=your-deepinfra-key-here
   
   # Optional: Custom paths
   # AURASEN_CONFIG_DIR=/path/to/custom/config
   # AURASEN_OUTPUT_DIR=/path/to/custom/output

Configuration Validation
-------------------------

AuraGen automatically validates configuration files on startup. Common validation errors:

API Key Issues
~~~~~~~~~~~~~~

.. code-block:: text

   ❌ Environment variable 'OPENAI_API_KEY' not set for api_key_type 'openai_api_key'
   
   Solution: Run python config/configure_api_keys.py

Invalid Model Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   ❌ Unknown model 'gpt-5' for OpenAI provider
   
   Solution: Check available models in the provider documentation

Missing Scenario Files
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   ❌ Scenario file not found: config/AgentSafetyBench/missing_scenario.yaml
   
   Solution: Ensure all referenced scenarios exist

Best Practices
--------------

Security
~~~~~~~~

* Store API keys in environment variables or ``.env`` files, never in YAML
* Use the ``api_key_type`` pattern for secure key resolution
* Add ``.env`` to your ``.gitignore`` file

Performance
~~~~~~~~~~~

* Adjust ``batch_size`` based on your API rate limits
* Use appropriate ``temperature`` values (higher for creativity, lower for consistency)
* Consider local models for high-volume generation

Organization
~~~~~~~~~~~~

* Keep scenario files organized by domain or use case
* Use descriptive names for custom API key types
* Document custom configurations with comments

Configuration Examples
----------------------

Development Setup
~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # config/generation.yaml - Development
   generation:
     batch_size: 5  # Smaller batches for testing
     externalAPI_generation: false
   
   openai:
     api_key_type: "openai_api_key"
     model: "gpt-3.5-turbo"  # Cheaper model for development
     temperature: 0.7
     max_tokens: 1024

Production Setup
~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # config/generation.yaml - Production
   generation:
     batch_size: 50  # Larger batches for efficiency
     externalAPI_generation: false
   
   openai:
     api_key_type: "openai_api_key"
     model: "gpt-4o"  # Best model for production
     temperature: 1.0
     max_tokens: 2048

Multi-Provider Setup
~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Use different providers for different purposes
   generation:
     batch_size: 20
     externalAPI_generation: true  # Use external for cost efficiency
   
   externalAPI:
     api_url: "https://api.deepinfra.com/v1/openai"
     api_key_type: "deepinfra_api_key"
     model: "Qwen/Qwen2.5-72B-Instruct"
     temperature: 1.0
     max_tokens: 4096

Troubleshooting Configuration
-----------------------------

For configuration issues, see the :doc:`advanced/troubleshooting` guide or check the logs for detailed error messages.

Next Steps
----------

* Learn about :doc:`scenarios` and how to customize them
* Explore :doc:`risk_injection` techniques
* Read about :doc:`advanced/api_integration` for custom providers
