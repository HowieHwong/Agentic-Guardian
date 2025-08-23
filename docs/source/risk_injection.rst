Risk Injection
==============

Risk injection transforms harmless AI trajectories into risky ones while preserving plausibility and coherence. AuraGen selects and applies risks based on scenario context and configuration.

Overview
--------

The risk injection system:

1. Analyzes harmless trajectories and scenario constraints
2. Selects a configured risk type and injection mode
3. Produces realistic risky variants while maintaining metadata

Configuration Source
--------------------

All risk categories and prompts are defined in `config/risk_injection.yaml`. Update that file to change behavior.

Risk Categories (from config/risk_injection.yaml)
-------------------------------------------------

Sensitive Data Privacy Violations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Name: ``sensitive_data_privacy_violations``
- Target: ``agent_action``
- Description: Unauthorized collection, use, disclosure, or leakage of sensitive, confidential, or personal information.
- Injection modes: ``single_action``, ``multiple_actions``, ``action_chain_with_response``, ``action_chain_only``

Property / Financial Loss
~~~~~~~~~~~~~~~~~~~~~~~~~

- Name: ``property_financial_loss``
- Target: ``agent_action``
- Description: Actions that directly or indirectly cause financial loss or damage to assets.
- Injection modes: ``single_action``, ``multiple_actions``, ``action_chain_with_response``, ``action_chain_only``

Misinformation / Unsafe Content
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Name: ``misinformation_unsafe_content``
- Target: ``agent_action``
- Description: Generation or dissemination of false, misleading, harmful, or unsafe information.
- Injection modes: ``single_action``, ``multiple_actions``, ``action_chain_with_response``, ``action_chain_only``

Compromised Availability
~~~~~~~~~~~~~~~~~~~~~~~~

- Name: ``compromised_availability``
- Target: ``agent_action``
- Description: Disruption or degradation of service availability, including resource exhaustion or denial of service.
- Injection modes: ``single_action``, ``multiple_actions``, ``action_chain_with_response``, ``action_chain_only``

Unintended / Unauthorized Actions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Name: ``unintended_unauthorized_actions``
- Target: ``agent_action``
- Description: Actions beyond user intent or permissions (e.g., privilege escalation, unauthorized automation).
- Injection modes: ``single_action``, ``multiple_actions``, ``action_chain_with_response``, ``action_chain_only``

External Adversarial Attack
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Name: ``external_adversarial_attack``
- Target: ``agent_action``
- Description: Susceptibility to prompt injection, jailbreaks, adversarial inputs, and other malicious manipulation.
- Injection modes: ``single_action``, ``multiple_actions``, ``action_chain_with_response``, ``action_chain_only``

Bias / Discrimination
~~~~~~~~~~~~~~~~~~~~~

- Name: ``bias_discrimination``
- Target: ``agent_action``
- Description: Outputs or decisions reflecting unfair bias, discrimination, or lack of fairness.
- Injection modes: ``single_action``, ``multiple_actions``, ``action_chain_with_response``, ``action_chain_only``

Lack of Accountability / Traceability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Name: ``lack_accountability_traceability``
- Target: ``agent_action``
- Description: Insufficient logging or explainability that impairs auditing or responsibility assignment.
- Injection modes: ``single_action``, ``multiple_actions``, ``action_chain_with_response``, ``action_chain_only``

Injection Modes
---------------

- ``single_action``: Modify a single step
- ``multiple_actions``: Modify multiple selected steps
- ``action_chain_with_response``: Modify a chain of actions and the response
- ``action_chain_only``: Modify the chain without changing the response

Basic Usage
-----------

.. code-block:: python

   from AuraGen.injection import RiskInjector
   from AuraGen.models import Trajectory
   from AuraGen.utils import load_yaml

   # Load configuration from YAML
   injector = RiskInjector.from_yaml("config/risk_injection.yaml")

   # Example harmless trajectory
   harmless = Trajectory(
       scenario_name="email_assistant",
       user_request="Draft an email to confirm tomorrow's meeting.",
       agent_action="compose_email",
       agent_response="Sure, I'll draft a professional confirmation email."
   )

   # Inject risk
   risky = injector.inject_risk(harmless)
   print(risky.metadata.get("risk_type"))

Manual vs. Automatic Target Selection
-------------------------------------

- Automatic: Set ``injection.auto_select_targets: true`` (default)
- Manual: Use entries in ``injection_configs`` with indices like ``target_indices`` or ``chain_start_index``

Outputs
-------

- Preserves original structure (request, action, response)
- Adds risk metadata (e.g., ``risk_type``, ``injection_mode``)
- Saved format controlled by ``output.file_format`` in ``config/risk_injection.yaml``
