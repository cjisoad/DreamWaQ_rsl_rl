# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class DWAQOnPolicyRunner(OnPolicyRunner):
    """Compatibility wrapper for DWAQ training.

    This class uses the DWAQ-aware code path implemented inside ``OnPolicyRunner``.
    """

    pass
