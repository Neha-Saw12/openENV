import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT))

from memory_engine import load_profile
from personality_grader import grade_purchase


class ServerProcess:
    def __init__(self, port: int) -> None:
        self.port = port
        self.process: subprocess.Popen[str] | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")]).strip(
            os.pathsep
        )
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "server.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
            ],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        for _ in range(30):
            try:
                response = requests.get(f"{self.base_url}/health", timeout=1)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                time.sleep(0.5)

        stderr = ""
        if self.process and self.process.stderr:
            stderr = self.process.stderr.read()
        raise RuntimeError(f"Server failed to start on port {self.port}: {stderr}")

    def stop(self) -> None:
        if not self.process:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


class OpenEnvRequirementsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ServerProcess(port=8010)
        cls.server.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.stop()

    def test_reset_uses_named_task_configuration(self) -> None:
        expected = {
            "quick_pick": ("lip balm", 4, 8),
            "smart_shop": ("earbuds", 8, 12),
            "expert_deal": ("laptop backpack", 12, 15),
        }

        for task_name, (query, product_count, max_steps) in expected.items():
            with self.subTest(task=task_name):
                response = requests.post(
                    f"{self.server.base_url}/reset",
                    json={"task": task_name},
                    timeout=10,
                )
                response.raise_for_status()
                payload = response.json()
                observation = payload["observation"]

                self.assertEqual(observation["query"], query)
                self.assertEqual(len(observation["candidate_products"]), product_count)
                self.assertEqual(observation["max_steps"], max_steps)

                state_response = requests.get(f"{self.server.base_url}/state", timeout=10)
                state_response.raise_for_status()
                state = state_response.json()
                self.assertEqual(state["step_count"], 0)

    def test_purchase_grader_is_bounded(self) -> None:
        profile = load_profile()
        products = [
            {
                "id": "p1",
                "name": "Budget Choice",
                "price": 20.0,
                "rating": 3.8,
                "brand": "ValueCo",
                "reviews": 120,
                "category": "earbuds",
                "seller": "Trusted Seller",
                "refundable": True,
                "archetype": "mid_range_best_value",
            },
            {
                "id": "p2",
                "name": "Risky Deal",
                "price": 8.0,
                "rating": 2.4,
                "brand": "Mystery",
                "reviews": 4,
                "category": "earbuds",
                "seller": "Unknown Flash Mart",
                "refundable": False,
                "archetype": "suspiciously_cheap",
            },
        ]

        score = grade_purchase(
            purchased_ids={"p1"},
            products=products,
            profile=profile,
            viewed=["p1", "p2"],
            compared_sets=[["p1", "p2"]],
            shortlisted=["p1"],
            skipped_ids=["p2"],
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_openenv_validate_passes(self) -> None:
        from openenv.cli._validation import validate_running_environment

        report = validate_running_environment(self.server.base_url, timeout_s=10)
        self.assertTrue(report["passed"], msg=json.dumps(report, indent=2))


if __name__ == "__main__":
    unittest.main()
