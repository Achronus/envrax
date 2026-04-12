from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import chex
import numpy as np

from envrax.env import ActSpaceT, JaxEnv, ObsSpaceT, StateT
from envrax.wrappers.base import Wrapper


class RecordVideo(Wrapper[ObsSpaceT, ActSpaceT, StateT]):
    """
    Save episode frames to MP4 based on configurable triggers.

    **Not JIT/vmap-compatible.** Intended for evaluation, logging, and
    training visualisation.

    Three optional triggers control when recording is active. They are
    OR'd together — if any trigger returns `True`, that episode is
    recorded. When no triggers are provided, every episode is recorded.

    Each completed recording is written to
    `<output_dir>/episode_<NNNN>.mp4` via `imageio`.

    Requires `imageio` with the `ffmpeg` plugin
    (`pip install "imageio[ffmpeg]"`).

    Parameters
    ----------
    env : JaxEnv
        Inner environment to wrap that has a `render()` method.
    output_dir : str | Path (optional)
        Directory where MP4 files are saved. Created automatically if
        it does not exist. Default is `runs/recordings`
    fps : int (optional)
        Frames per second for the saved video. Default is `30`.
    episode_trigger : Callable[[int], bool] (optional)
        Called with the episode count at each `reset()`. If `True`,
        record this episode. Useful for "record every Nth episode".
        Default is `None`
    step_trigger : Callable[[int], bool] (optional)
        Called with the global step count at each `step()`. If `True`,
        start recording from this step until the episode ends.
        Default is `None`
    recording_trigger : Callable[[], bool] (optional)
        Zero-arg callable checked at each `reset()`. If `True`, record
        this episode. Useful for meta-learning where the framework
        controls when to record via an external flag.
        Default is `None`

    Raises
    ------
    render_missing : TypeError
        If the unwrapped environment does not implement `render()`.
    """

    def __init__(
        self,
        env: JaxEnv[ObsSpaceT, ActSpaceT, StateT],
        *,
        output_dir: str | Path = "runs/recordings",
        fps: int = 30,
        episode_trigger: Callable[[int], bool] | None = None,
        step_trigger: Callable[[int], bool] | None = None,
        recording_trigger: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(env)

        # Fail fast if the env doesn't support rendering
        if type(self.unwrapped).render is JaxEnv.render:
            raise TypeError(
                f"RecordVideo requires an environment that implements render(). "
                f"{type(self.unwrapped).__name__} does not."
            )

        self.output_dir = Path(output_dir)
        self.fps = fps
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._episode_trigger = episode_trigger
        self._step_trigger = step_trigger
        self._recording_trigger = recording_trigger
        self._has_triggers = any(
            t is not None for t in (episode_trigger, step_trigger, recording_trigger)
        )

        self._frames: List[np.ndarray] = []
        self._episode_id: int = 0
        self._global_step: int = 0
        self._recording: bool = False

    @property
    def recording(self) -> bool:
        """Whether the current episode is being recorded."""
        return self._recording

    def _should_record_episode(self) -> bool:
        """Check episode-level triggers (OR logic)."""
        if not self._has_triggers:
            return True  # no triggers → record everything

        if self._episode_trigger is not None and self._episode_trigger(
            self._episode_id
        ):
            return True

        if self._recording_trigger is not None and self._recording_trigger():
            return True

        return False

    def reset(self, rng: chex.PRNGKey) -> Tuple[chex.Array, StateT]:
        """
        Reset the environment and optionally begin a new recording.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key

        Returns
        -------
        obs : chex.Array
            First observation
        state : StateT
            Initial environment state
        """
        obs, state = self._env.reset(rng)

        self._recording = self._should_record_episode()

        if self._recording:
            self._frames = [np.asarray(self._env.render(state))]
        else:
            self._frames = []

        return obs, state

    def step(
        self,
        state: StateT,
        action: chex.Array,
    ) -> Tuple[chex.Array, StateT, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Advance the environment by one step and record the frame if active.

        If a `step_trigger` is provided and fires, recording starts
        mid-episode and continues until the episode ends.

        Flushes accumulated frames to an MP4 file when `done` is `True`.

        Parameters
        ----------
        state : StateT
            Current environment state
        action : chex.Array
            Action to take in the environment

        Returns
        -------
        obs : chex.Array
            Observation after the step
        new_state : StateT
            Updated environment state
        reward : chex.Array
            Reward for this step
        done : chex.Array
            `True` when the episode has ended
        info : Dict[str, Any]
            Pass-through info dict from the inner environment
        """
        obs, new_state, reward, done, info = self._env.step(state, action)
        self._global_step += 1

        # Mid-episode trigger: start recording if step_trigger fires
        if (
            not self._recording
            and self._step_trigger is not None
            and self._step_trigger(self._global_step)
        ):
            self._recording = True

        if self._recording:
            self._frames.append(np.asarray(self._env.render(new_state)))

        if bool(done) and self._recording:
            self._flush()
            self._recording = False

        if bool(done):
            self._episode_id += 1

        return obs, new_state, reward, done, info

    def _flush(self) -> None:
        """Write accumulated frames to an MP4 file."""
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
        frames: list[Any] = self._frames
        imageio.mimwrite(str(path), frames, fps=self.fps)
        self._frames = []
