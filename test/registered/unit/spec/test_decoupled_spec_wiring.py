"""Unit tests for phase-one decoupled speculative role wiring."""

import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.speculative_hook import _handle_decoupled_spec
from sglang.srt.environ import envs
from sglang.srt.managers.utils import compute_num_reserved_tokens
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDecoupledSpecRoleWiring(CustomTestCase):
    def setUp(self) -> None:
        cpp_data_plane_override = envs.SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND.override(
            True
        )
        cpp_data_plane_override.__enter__()
        self.addCleanup(cpp_data_plane_override.__exit__, None, None, None)

    @staticmethod
    def _args(role: str, **overrides):
        values = dict(
            decoupled_spec_role=role,
            speculative_algorithm=("DECOUPLED_VERIFY" if role == "verifier" else None),
            disable_overlap_schedule=(role == "drafter"),
            dp_size=1,
            pp_size=1,
            tp_size=1,
            decoupled_spec_bind_endpoint="ipc:///tmp/decoupled-local",
            decoupled_spec_connect_endpoints=["ipc:///tmp/decoupled-peer"],
            decoupled_spec_rank=0,
            speculative_eagle_topk=None,
            speculative_num_steps=3,
            speculative_num_draft_tokens=None,
            page_size=(1 if role == "drafter" else 64),
            mamba_radix_cache_strategy="extra_buffer",
            enable_linear_replayssm=False,
            max_running_requests=None,
            max_mamba_cache_size=None,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_verifier_keeps_overlap_and_official_cache_geometry(self):
        args = self._args("verifier", disable_overlap_schedule=False)

        _handle_decoupled_spec(args)

        self.assertFalse(args.disable_overlap_schedule)
        self.assertEqual(args.page_size, 64)
        self.assertEqual(args.mamba_radix_cache_strategy, "extra_buffer")
        self.assertEqual(args.speculative_eagle_topk, 1)
        self.assertEqual(args.speculative_num_draft_tokens, 4)

    def test_verifier_reserves_the_full_target_verify_width(self):
        args = self._args("verifier")
        _handle_decoupled_spec(args)
        args.max_speculative_num_draft_tokens = args.speculative_num_draft_tokens

        self.assertEqual(compute_num_reserved_tokens(args), 4)

    def test_drafter_is_plain_non_overlap_decode(self):
        args = self._args("drafter")

        _handle_decoupled_spec(args)

        self.assertIsNone(args.speculative_algorithm)
        self.assertTrue(args.disable_overlap_schedule)
        self.assertEqual(args.speculative_eagle_topk, 1)
        self.assertEqual(args.speculative_num_draft_tokens, 4)
        self.assertEqual(args.max_running_requests, 1)
        self.assertEqual(args.max_mamba_cache_size, 8)

    def test_drafter_rollback_geometry_is_fail_fast(self):
        cases = [
            (dict(tp_size=2), "tp_size == 1"),
            (dict(page_size=64), "page_size == 1"),
            (dict(enable_linear_replayssm=True), "ReplaySSM disabled"),
            (
                dict(max_running_requests=2, max_mamba_cache_size=15),
                "required=16",
            ),
        ]
        for overrides, error in cases:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(
                ValueError, error
            ):
                _handle_decoupled_spec(self._args("drafter", **overrides))

    def test_role_algorithm_contract_is_fail_fast(self):
        cases = [
            (
                "verifier_without_builtin",
                self._args("verifier", speculative_algorithm=None),
                "requires --speculative-algorithm DECOUPLED_VERIFY",
            ),
            (
                "drafter_with_algorithm",
                self._args("drafter", speculative_algorithm="EAGLE"),
                "plain decode engine",
            ),
            (
                "builtin_without_role",
                self._args("null", speculative_algorithm="DECOUPLED_VERIFY"),
                "requires --decoupled-spec-role verifier",
            ),
            (
                "overlap_drafter",
                self._args("drafter", disable_overlap_schedule=False),
                "requires --disable-overlap-schedule",
            ),
        ]
        for name, args, error in cases:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, error):
                _handle_decoupled_spec(args)

    def test_active_runtime_requires_the_cpp_data_plane(self):
        for role in ("verifier", "drafter"):
            with self.subTest(
                role=role
            ), envs.SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND.override(
                False
            ), self.assertRaisesRegex(
                ValueError, "requires the native C\\+\\+ data plane"
            ):
                _handle_decoupled_spec(self._args(role))

    def test_phase_one_topology_is_one_to_one(self):
        cases = [
            ("dp", dict(dp_size=2), "dp_size == 1"),
            ("pp", dict(pp_size=2), "pp_size == 1"),
            (
                "peers",
                dict(decoupled_spec_connect_endpoints=["ipc:///tmp/a", "ipc:///tmp/b"]),
                "exactly one peer",
            ),
            ("rank", dict(decoupled_spec_rank=1), "requires --decoupled-spec-rank 0"),
        ]
        for name, overrides, error in cases:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, error):
                _handle_decoupled_spec(self._args("verifier", **overrides))

    def test_linear_k_relation_is_fail_fast(self):
        cases = [
            (dict(speculative_eagle_topk=2), "topk 1"),
            (dict(speculative_num_steps=0), "positive"),
            (dict(speculative_num_draft_tokens=3), "must equal"),
        ]
        for overrides, error in cases:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(
                ValueError, error
            ):
                _handle_decoupled_spec(self._args("verifier", **overrides))


if __name__ == "__main__":
    unittest.main()
