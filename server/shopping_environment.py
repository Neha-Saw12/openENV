"""
OpenEnv-compatible ShoppingEnvironment.

Wraps the existing shopping_env logic into the OpenEnv Environment
interface (reset → Observation, step → Observation, state → property).
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

# Ensure parent directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import (
    EnvironmentMetadata,
    State,
)

from openenv_models import ShoppingAction, ShoppingObservation, ShoppingState
from memory_engine import load_profile, UserProfile
from product_generator import generate_products
from personality_grader import grade_purchase as personality_grade_purchase, score_all_products
from task_config import DEFAULT_TASK_NAME, get_task_config


DEFAULT_MAX_STEPS = 15
DEFAULT_QUERY = "earbuds"


class ShoppingEnvironment(Environment[ShoppingAction, ShoppingObservation, ShoppingState]):
    """
    OpenEnv-compliant shopping environment.

    Implements the Environment interface:
      - reset()  → ShoppingObservation
      - step()   → ShoppingObservation
      - state    → ShoppingState (property)
      - close()  → cleanup
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._user_profile: UserProfile = load_profile()

        # Episode state
        self._step_count = 0
        self._max_steps = DEFAULT_MAX_STEPS
        self._done = False
        self._cumulative_reward = 0.0
        self._query = ""
        self._category = ""
        self._cart: List[str] = []
        self._shortlisted: List[str] = []
        self._viewed: List[str] = []
        self._compared_sets: List[List[str]] = []
        self._history: List[str] = []
        self._feedback = ""
        self._skipped_ids: List[str] = []
        self.catalog: List[Dict[str, Any]] = []
        self._scored_products: List[Dict[str, Any]] = []
        self._episode_id = str(uuid4())
        self._task_name = DEFAULT_TASK_NAME
        self._difficulty = "medium"

    # ---- OpenEnv interface -------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        query: str = DEFAULT_QUERY,
        product_count: int = 8,
        task_name: str = DEFAULT_TASK_NAME,
        max_steps: int = DEFAULT_MAX_STEPS,
        **kwargs: Any,
    ) -> ShoppingObservation:
        """Reset the environment. Returns initial ShoppingObservation."""
        requested_task = kwargs.get("task", task_name)
        task_config = get_task_config(requested_task)
        query_was_explicit = "query" in kwargs or query != DEFAULT_QUERY
        product_count_was_explicit = (
            "product_count" in kwargs or product_count != 8
        )
        max_steps_was_explicit = "max_steps" in kwargs or max_steps != DEFAULT_MAX_STEPS

        if task_config:
            if not query_was_explicit:
                query = task_config.get("query", query)
            if not product_count_was_explicit:
                product_count = task_config.get("product_count", product_count)
            if not max_steps_was_explicit:
                max_steps = task_config.get("max_steps", max_steps)
            task_name = task_config["name"]
            self._difficulty = task_config.get("difficulty", task_name)
        else:
            self._difficulty = "dynamic"

        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._query = query.strip() or DEFAULT_QUERY
        self._category = self._query.lower()
        self._cart = []
        self._shortlisted = []
        self._viewed = []
        self._compared_sets = []
        self._history = []
        self._skipped_ids = []
        self._episode_id = episode_id or str(uuid4())
        self._task_name = task_name
        self._max_steps = max_steps

        # Reload user profile (may have been updated)
        self._user_profile = load_profile()

        # Generate products dynamically
        self.catalog = generate_products(self._query, count=product_count)

        # Score all products against personality
        self._scored_products = score_all_products(self.catalog, self._user_profile)

        # Find the personality-ideal product
        ideal = self._scored_products[0] if self._scored_products else None
        ideal_name = ideal["product"]["name"] if ideal else "unknown"
        ideal_score = ideal["personality_score"] if ideal else 0
        research_depth = self._user_profile.research_depth

        self._feedback = (
            f"Welcome! Shopping for: {self._query}\n"
            f"Generated {len(self.catalog)} products.\n"
            f"Goal: Find the product that best matches your personality profile.\n"
            f"The personality-ideal product is '{ideal_name}' "
            f"(alignment score: {ideal_score:.2f}).\n"
            f"Your research depth preference: {research_depth:.0%} — "
            f"{'thorough research expected' if research_depth > 0.7 else 'quick decisions OK'}."
        )

        return self._get_obs(reward=0.0)

    def step(
        self,
        action: ShoppingAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ShoppingObservation:
        """Execute an action. Returns ShoppingObservation with reward and done."""
        if self._done:
            return self._get_obs(reward=0.0, done=True)

        self._step_count += 1
        reward = 0.0
        action_log = ""
        research_depth = self._user_profile.research_depth
        atype = action.action_type.lower().strip()

        # ---- search -------------------------------------------------------
        if atype == "search":
            self._query = action.search_query or self._query
            action_log = f"Searched for '{self._query}'"
            self._feedback = f"Found {len(self.catalog)} products for '{self._query}'."
            reward = 0.05 * research_depth

        # ---- view_item ----------------------------------------------------
        elif atype == "view_item":
            if action.item_ids:
                details = []
                for pid in action.item_ids:
                    if pid not in self._viewed:
                        self._viewed.append(pid)
                    prod = self._find_product(pid)
                    if prod:
                        ps = self._get_personality_score(pid)
                        details.append(
                            f"{prod['name']}: ${prod['price']}, "
                            f"{prod['rating']}★, {prod['reviews']} reviews, "
                            f"brand {prod['brand']}, seller {prod['seller']}, "
                            f"refundable={prod['refundable']}, "
                            f"personality_alignment={ps:.2f}"
                        )
                action_log = f"Viewed {len(action.item_ids)} item(s): {action.item_ids}"
                self._feedback = (
                    "Details:\n" + "\n".join(f"  - {d}" for d in details)
                    if details else "No matching products found."
                )
                reward = 0.05 * min(len(action.item_ids), 4) * research_depth
            else:
                self._feedback = "view_item requires at least one item_id."

        # ---- compare ------------------------------------------------------
        elif atype == "compare":
            if not action.item_ids or len(action.item_ids) < 2:
                self._feedback = "Compare requires at least 2 item_ids."
            else:
                self._compared_sets.append(list(action.item_ids))
                names = []
                for pid in action.item_ids:
                    p = self._find_product(pid)
                    if p:
                        ps = self._get_personality_score(pid)
                        names.append(
                            f"{p['name']} (${p['price']}, {p['rating']}★, "
                            f"alignment: {ps:.2f})"
                        )
                action_log = f"Compared {len(action.item_ids)} items: {action.item_ids}"
                self._feedback = "Comparison:\n" + "\n".join(f"  - {n}" for n in names)
                reward = 0.05 * min(len(action.item_ids), 5) * research_depth

        # ---- shortlist ----------------------------------------------------
        elif atype == "shortlist":
            for pid in action.item_ids:
                if pid not in self._shortlisted:
                    self._shortlisted.append(pid)
            action_log = f"Shortlisted {action.item_ids}"
            self._feedback = f"Shortlist now: {self._shortlisted}"
            reward = 0.1 * research_depth

        # ---- add_to_cart --------------------------------------------------
        elif atype == "add_to_cart":
            for pid in action.item_ids:
                if pid not in self._cart:
                    self._cart.append(pid)
            action_log = f"Added to cart: {action.item_ids}"
            self._feedback = f"Cart: {self._cart}"
            reward = 0.1

        # ---- remove_from_cart ---------------------------------------------
        elif atype == "remove_from_cart":
            for pid in action.item_ids:
                if pid in self._cart:
                    self._cart.remove(pid)
            action_log = f"Removed from cart: {action.item_ids}"
            self._feedback = f"Cart: {self._cart}"
            reward = 0.0

        # ---- buy ----------------------------------------------------------
        elif atype == "buy":
            action_log = "Attempted purchase"
            self._done = True
            purchased = set(self._cart + action.item_ids)
            if not purchased:
                self._feedback = "Cannot buy — cart is empty and no item_ids given."
                reward = 0.0
            else:
                reward = self._grade_purchase(purchased)
                purchased_names = []
                for pid in purchased:
                    p = self._find_product(pid)
                    if p:
                        ps = self._get_personality_score(pid)
                        purchased_names.append(f"{p['name']} (alignment: {ps:.2f})")
                self._feedback = (
                    f"Purchased: {', '.join(purchased_names)}.\n"
                    f"Final score: {reward:.2f}"
                )
                self._log_episode(purchased, reward)

        # ---- skip ---------------------------------------------------------
        elif atype == "skip":
            if action.item_ids:
                self._skipped_ids.extend(action.item_ids)
                action_log = f"Skipped items: {action.item_ids}"
            else:
                action_log = "Skipped turn"
            self._feedback = "Turn skipped."
            reward = 0.0

        # ---- ask_more -----------------------------------------------------
        elif atype == "ask_more":
            action_log = "Asked for more options"
            self._feedback = "No additional products available in this catalog."
            reward = 0.0

        # ---- unknown ------------------------------------------------------
        else:
            action_log = f"Unknown action: {atype}"
            self._feedback = (
                f"Invalid action_type '{atype}'. Valid: search, view_item, "
                f"compare, shortlist, add_to_cart, remove_from_cart, buy, skip, ask_more."
            )
            reward = -0.1

        if action_log:
            self._history.append(action_log)

        # Penalize running out of steps
        if self._step_count >= self._max_steps and not self._done:
            self._done = True
            self._feedback += " Episode ended — max steps reached without purchase."
            reward -= 0.2

        self._cumulative_reward += reward
        return self._get_obs(reward=round(reward, 4), done=self._done)

    @property
    def state(self) -> ShoppingState:
        """Get current episode state."""
        return ShoppingState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_name=self._task_name,
            difficulty=self._difficulty,
            done=self._done,
            cumulative_reward=round(self._cumulative_reward, 4),
            cart=list(self._cart),
            shortlisted=list(self._shortlisted),
            product_query=self._query,
        )

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="shopping_agent",
            description=(
                "Personality-driven RL shopping environment. "
                "The agent learns to shop like a specific user."
            ),
            version="1.0.0",
        )

    def close(self) -> None:
        pass

    # ---- helpers ----------------------------------------------------------

    def _find_product(self, pid: str) -> Optional[Dict[str, Any]]:
        for p in self.catalog:
            if p["id"] == pid:
                return p
        return None

    def _get_personality_score(self, pid: str) -> float:
        for item in self._scored_products:
            if item["product"]["id"] == pid:
                return item["personality_score"]
        return 0.0

    def _get_obs(self, reward: float = 0.0, done: bool = False) -> ShoppingObservation:
        history_text = "\n".join(self._history[-6:]) if self._history else "No actions yet."
        profile = self._user_profile
        prefs = profile.get_prefs_for_category(self._category)

        return ShoppingObservation(
            done=done,
            reward=reward,
            query=self._query,
            category=self._category,
            candidate_products=self.catalog,
            memory_profile={
                "goal": (
                    f"Find the best {self._query} that matches your personality: "
                    f"research-heavy ({prefs.get('research_depth', 0.5):.0%}), "
                    f"value-conscious ({prefs.get('price_sensitivity', 0.5):.0%}), "
                    f"quality-focused ({prefs.get('quality_preference', 0.5):.0%})."
                ),
                **prefs,
                "semantic_conclusions": [
                    c.get("conclusion", "")
                    for c in profile.semantic_conclusions[:6]
                ],
                "personality_summary": profile.personality_summary[:300],
            },
            cart=list(self._cart),
            shortlisted=list(self._shortlisted),
            viewed_items=list(self._viewed),
            compared_sets=[list(s) for s in self._compared_sets],
            history_summary=history_text,
            feedback=self._feedback,
            step_number=self._step_count,
            max_steps=self._max_steps,
        )

    def _grade_purchase(self, purchased_ids: set) -> float:
        return personality_grade_purchase(
            purchased_ids=purchased_ids,
            products=self.catalog,
            profile=self._user_profile,
            viewed=self._viewed,
            compared_sets=self._compared_sets,
            shortlisted=self._shortlisted,
            skipped_ids=self._skipped_ids,
        )

    def _log_episode(self, purchased_ids: set, reward: float):
        try:
            import datetime
            log_path = Path(__file__).parent.parent / "memory" / "episodic_log.jsonl"
            purchased_products = []
            for pid in purchased_ids:
                p = self._find_product(pid)
                if p:
                    purchased_products.append({
                        "id": p["id"],
                        "name": p["name"],
                        "price": p["price"],
                        "brand": p["brand"],
                    })
            entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "event": "agent_purchase",
                "query": self._query,
                "purchased": purchased_products,
                "reward": round(reward, 4),
                "steps": self._step_count,
                "viewed_count": len(self._viewed),
                "compared_count": len(self._compared_sets),
                "shortlisted_count": len(self._shortlisted),
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[shopping_env] Error logging episode: {e}")
