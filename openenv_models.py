"""
OpenEnv-compatible models for the Shopping Agent Environment.

These models inherit from openenv's base Action/Observation/State types
so they can be used with create_app() and EnvClient.
"""

from typing import Any, Dict, List, Optional

from pydantic import Field

from openenv.core.env_server.types import (
    Action as OpenEnvAction,
    Observation as OpenEnvObservation,
    State as OpenEnvState,
)


# ---------------------------------------------------------------------------
# Action: what the agent sends (OpenEnv-compatible)
# ---------------------------------------------------------------------------
class ShoppingAction(OpenEnvAction):
    """
    An action the agent can take in the shopping environment.

    Supported action_types:
      search, view_item, compare, shortlist, add_to_cart,
      remove_from_cart, buy, skip, ask_more
    """

    action_type: str = Field(
        ...,
        description="One of: search, view_item, compare, shortlist, "
                    "add_to_cart, remove_from_cart, buy, skip, ask_more",
    )
    item_ids: List[str] = Field(
        default_factory=list,
        description="Product IDs involved in the action",
    )
    search_query: Optional[str] = Field(
        default=None,
        description="Query string when action_type is 'search'",
    )

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Observation: what the environment returns (OpenEnv-compatible)
# ---------------------------------------------------------------------------
class ShoppingObservation(OpenEnvObservation):
    """What the agent observes after each step — OpenEnv Observation subclass."""

    query: str = Field("", description="Current search query")
    category: str = Field("", description="Current product category")
    candidate_products: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Products currently visible to the agent",
    )
    memory_profile: Dict[str, Any] = Field(
        default_factory=dict,
        description="User personality traits and semantic memory",
    )
    cart: List[str] = Field(default_factory=list, description="Product IDs in cart")
    shortlisted: List[str] = Field(default_factory=list, description="Shortlisted IDs")
    viewed_items: List[str] = Field(default_factory=list, description="Viewed IDs")
    compared_sets: List[List[str]] = Field(
        default_factory=list, description="Compared product ID sets"
    )
    history_summary: str = Field("", description="Recent actions summary")
    feedback: str = Field("", description="Environment feedback")
    step_number: int = Field(0, description="Current step")
    max_steps: int = Field(15, description="Max steps")

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# State: episode metadata (OpenEnv-compatible)
# ---------------------------------------------------------------------------
class ShoppingState(OpenEnvState):
    """Episode-level metadata — OpenEnv State subclass."""

    task_name: str = ""
    difficulty: str = ""
    done: bool = False
    cumulative_reward: float = 0.0
    cart: List[str] = Field(default_factory=list)
    shortlisted: List[str] = Field(default_factory=list)
    product_query: str = Field(default="", description="Current product query")
