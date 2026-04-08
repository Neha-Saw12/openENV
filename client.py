"""
OpenEnv Client for the Shopping Agent Environment.

Connects to the shopping environment server via WebSocket (OpenEnv protocol).
"""

from typing import Any, Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from openenv_models import ShoppingAction, ShoppingObservation, ShoppingState


class ShoppingEnvClient(EnvClient[ShoppingAction, ShoppingObservation, ShoppingState]):
    """
    WebSocket client for the Shopping Agent Environment.

    Usage (async):
        async with ShoppingEnvClient(base_url="http://localhost:8000") as env:
            result = await env.reset(query="lip balm", product_count=4)
            result = await env.step(ShoppingAction(action_type="view_item", item_ids=["p1"]))

    Usage (sync):
        with ShoppingEnvClient(base_url="http://localhost:8000").sync() as env:
            result = env.reset(query="earbuds")
            result = env.step(ShoppingAction(action_type="buy", item_ids=["p1"]))
    """

    def _step_payload(self, action: ShoppingAction) -> Dict[str, Any]:
        """Convert ShoppingAction to the JSON payload expected by the server."""
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ShoppingObservation]:
        """Parse server response into StepResult[ShoppingObservation]."""
        obs_data = payload.get("observation", payload)
        reward = payload.get("reward") or obs_data.get("reward", 0.0)
        done = payload.get("done", obs_data.get("done", False))

        observation = ShoppingObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> ShoppingState:
        """Parse server state response into ShoppingState."""
        return ShoppingState(**payload)
