"""
Defines different modes of risk injection.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class InjectionMode(str, Enum):
    """Enum for different risk injection modes."""
    SINGLE_ACTION = "single_action"  # Modify a single action
    MULTIPLE_ACTIONS = "multiple_actions"  # Modify multiple selected actions
    ACTION_CHAIN_WITH_RESPONSE = "action_chain_with_response"  # Modify action chain and response
    ACTION_CHAIN_ONLY = "action_chain_only"  # Modify action chain only


class InjectionConfig(BaseModel):
    """Configuration for risk injection."""
    mode: InjectionMode = Field(
        InjectionMode.SINGLE_ACTION,
        description="The mode of injection to use"
    )
    target_indices: Optional[List[int]] = Field(
        None,
        description="Indices of actions to modify. If None, model will choose automatically."
    )
    chain_start_index: Optional[int] = Field(
        None,
        description="Starting index for chain modification. Used in chain modes only."
    )
    modify_response: bool = Field(
        False,
        description="Whether to modify the agent response"
    )
    auto_select_targets: bool = Field(
        True,
        description="Whether to let the model automatically select target indices"
    )

    def validate_config(self) -> List[str]:
        """Validate the injection configuration."""
        errors = []
        
        if not self.auto_select_targets:
            if self.target_indices is None and self.mode != InjectionMode.ACTION_CHAIN_WITH_RESPONSE and self.mode != InjectionMode.ACTION_CHAIN_ONLY:
                errors.append("target_indices cannot be None when auto_select_targets is False")
            
            if self.mode == InjectionMode.SINGLE_ACTION and self.target_indices and len(self.target_indices) > 1:
                errors.append("SINGLE_ACTION mode can only have one target index")
        
        if self.mode in [InjectionMode.ACTION_CHAIN_WITH_RESPONSE, InjectionMode.ACTION_CHAIN_ONLY]:
            if not self.auto_select_targets and self.chain_start_index is None:
                errors.append("chain_start_index is required for chain modification modes when auto_select_targets is False")
        
        if self.mode == InjectionMode.ACTION_CHAIN_WITH_RESPONSE:
            if not self.modify_response:
                errors.append("modify_response must be True for ACTION_CHAIN_WITH_RESPONSE mode")
        
        return errors 