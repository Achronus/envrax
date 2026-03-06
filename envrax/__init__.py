# Copyright 2026 Achronus
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from envrax.base import EnvParams, EnvState, JaxEnv
from envrax.envs import EnvGroup, EnvSet
from envrax.error import MissingPackageError
from envrax.make import make, make_multi, make_multi_vec, make_vec
from envrax.registry import make_env, register, registered_names
from envrax.spaces import Box, Discrete, Space
from envrax.wrappers import (
    ClipReward,
    EpisodeDiscount,
    EpisodeStatisticsState,
    ExpandDims,
    FrameStackObservation,
    FrameStackState,
    GrayscaleObservation,
    JitWrapper,
    NormalizeObservation,
    RecordEpisodeStatistics,
    RecordVideo,
    ResizeObservation,
    VmapEnv,
    Wrapper,
)

__all__ = [
    "Box",
    "ClipReward",
    "Discrete",
    "EnvGroup",
    "EnvParams",
    "EnvSet",
    "EnvState",
    "EpisodeDiscount",
    "MissingPackageError",
    "EpisodeStatisticsState",
    "ExpandDims",
    "FrameStackObservation",
    "FrameStackState",
    "GrayscaleObservation",
    "JaxEnv",
    "JitWrapper",
    "NormalizeObservation",
    "RecordEpisodeStatistics",
    "RecordVideo",
    "ResizeObservation",
    "Space",
    "VmapEnv",
    "Wrapper",
    "make",
    "make_env",
    "make_multi",
    "make_multi_vec",
    "make_vec",
    "register",
    "registered_names",
]
