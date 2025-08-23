Installation
============

AuraGen can be set up quickly using Conda. This guide provides a concise, conda-only installation path.

Conda Setup (Recommended)
-------------------------

.. code-block:: bash

   # Create and activate environment
   conda create -n guardian python=3.11 -y
   conda activate guardian

   # Install project dependencies
   pip install -r requirements.txt

Verify Installation
-------------------

.. code-block:: bash

   python -c "from AuraGen import core; print('AuraGen installed successfully!')"

Next Steps
----------

- Proceed to :doc:`quickstart` to run your first generation and risk injection
- See :doc:`configuration` for API key setup and configuration details
