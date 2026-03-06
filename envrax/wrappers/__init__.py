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

from envrax.wrappers.base import Wrapper, _WrapperFactory
from envrax.wrappers.clip_reward import ClipReward
from envrax.wrappers.discount import EpisodeDiscount
from envrax.wrappers.expand_dims import ExpandDims
from envrax.wrappers.frame_stack import FrameStackObservation, FrameStackState
from envrax.wrappers.grayscale import GrayscaleObservation
from envrax.wrappers.jit_wrapper import JitWrapper
from envrax.wrappers.normalize_obs import NormalizeObservation
from envrax.wrappers.record_episode_statistics import (
    EpisodeStatisticsState,
    RecordEpisodeStatistics,
)
from envrax.wrappers.record_video import RecordVideo
from envrax.wrappers.resize import ResizeObservation
from envrax.wrappers.utils import resize, to_gray
from envrax.wrappers.vmap_env import VmapEnv

__all__ = [
    "ClipReward",
    "EpisodeDiscount",
    "EpisodeStatisticsState",
    "ExpandDims",
    "FrameStackObservation",
    "FrameStackState",
    "GrayscaleObservation",
    "JitWrapper",
    "NormalizeObservation",
    "RecordEpisodeStatistics",
    "RecordVideo",
    "ResizeObservation",
    "VmapEnv",
    "Wrapper",
    "_WrapperFactory",
    "resize",
    "to_gray",
]
