from pathlib import Path
from typing import Any, Dict, List, Tuple

import chex
import numpy as np

from envrax.base import EnvConfig, JaxEnv
from envrax.wrappers.base import Wrapper


class RecordVideo(Wrapper):
    """Save episode frames to MP4 when each episode ends.

    **Not JIT/vmap-compatible.** Intended for evaluation runs and
    interactive play, not compiled training loops.

    Each completed episode is written to
    `<output_dir>/episode_<NNNN>.mp4` via `imageio`.

    Requires `imageio` with the `ffmpeg` plugin
    (`pip install "imageio[ffmpeg]"`).

    Parameters
    ----------
    env : JaxEnv
        Inner environment to wrap.
    output_dir : str or Path
        Directory where MP4 files are saved. Created automatically if
        it does not exist.
    fps : int (optional)
        Frames per second for the saved video. Default is ``30``.
    """

    def __init__(self, env: JaxEnv, output_dir: str | Path, fps: int = 30) -> None:
        super().__init__(env)
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._frames: List[np.ndarray] = []
        self._episode_id: int = 0

    def reset(self, rng: chex.PRNGKey, config: EnvConfig) -> Tuple[chex.Array, Any]:
        """
        Reset the environment and begin a new recording.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key.
        config : EnvConfig
            Environment configuration.

        Returns
        -------
        obs  : chex.Array
            First observation.
        state : Any
            Initial environment state.
        """
        obs, state = self._env.reset(rng, config)
        inner = self.unwrapped
        if hasattr(inner, "render"):
            self._frames = [np.asarray(inner.render(state))]
        else:
            self._frames = []
        return obs, state

    def step(
        self,
        rng: chex.PRNGKey,
        state: Any,
        action: chex.Array,
        config: EnvConfig,
    ) -> Tuple[chex.Array, Any, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Advance the environment by one step and record the frame.

        Flushes the accumulated frames to an MP4 file when ``done`` is True.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key.
        state : Any
            Current environment state.
        action : chex.Array
            int32 — Action index.
        config : EnvConfig
            Environment configuration.

        Returns
        -------
        obs  : chex.Array
            Observation after the step.
        new_state : Any
            Updated environment state.
        reward  : chex.Array
            float32 — Reward for this step.
        done  : chex.Array
            bool — True when the episode has ended.
        info : Dict[str, Any]
            Pass-through info dict from the inner environment.
        """
        obs, new_state, reward, done, info = self._env.step(rng, state, action, config)
        inner = self.unwrapped
        if hasattr(inner, "render"):
            self._frames.append(np.asarray(inner.render(new_state)))
        if bool(done):
            self._flush()
        return obs, new_state, reward, done, info

    def _flush(self) -> None:
        if not self._frames:
            return
        try:
            import imageio
        except ImportError as exc:
            raise ImportError(
                "imageio is required for video recording. "
                'Install it with: pip install "imageio[ffmpeg]"'
            ) from exc

        path = self.output_dir / f"episode_{self._episode_id:04d}.mp4"
        imageio.mimsave(str(path), self._frames, fps=self.fps)
        self._episode_id += 1
        self._frames = []
