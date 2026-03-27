"""Synthesized sound effects for ice hockey — no asset files needed."""

from __future__ import annotations

import numpy as np

from games.hockey.constants import (
    MASTER_VOLUME,
    SAMPLE_RATE,
    SFX_VOLUME,
    SOUND_ENABLED,
)

# Pygame import is deferred so modules can be imported without pygame init
_mixer_initialized = False
_sounds: dict[str, object] = {}
_muted = False


def init() -> None:
    """Initialize the sound system and generate all sounds."""
    global _mixer_initialized
    if not SOUND_ENABLED:
        return

    import pygame.mixer

    try:
        pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
        pygame.mixer.set_num_channels(16)
        _mixer_initialized = True
    except Exception:
        return

    _generate_all_sounds()


def _generate_all_sounds() -> None:
    """Generate all sound effects from numpy arrays."""
    import pygame.sndarray

    sounds = {
        "shot": _synth_shot(),
        "pass": _synth_pass(),
        "wall_bounce": _synth_wall_bounce(),
        "hit": _synth_body_check(),
        "pickup": _synth_pickup(),
        "goal": _synth_goal_horn(),
        "whistle": _synth_whistle(),
        "buzzer": _synth_buzzer(),
    }

    for name, samples in sounds.items():
        # Ensure stereo int16
        if samples.ndim == 1:
            samples = np.column_stack([samples, samples])
        samples = np.clip(samples, -32767, 32767).astype(np.int16)
        # Make contiguous C-array for pygame
        samples = np.ascontiguousarray(samples)
        snd = pygame.sndarray.make_sound(samples)
        snd.set_volume(SFX_VOLUME * MASTER_VOLUME)
        _sounds[name] = snd


def play(event_name: str) -> None:
    """Play a sound effect by event name."""
    if not _mixer_initialized or _muted:
        return
    snd = _sounds.get(event_name)
    if snd is not None:
        snd.play()


def toggle_mute() -> bool:
    """Toggle mute. Returns new mute state."""
    global _muted
    _muted = not _muted
    return _muted


def cleanup() -> None:
    """Shut down mixer."""
    if _mixer_initialized:
        import pygame.mixer
        pygame.mixer.quit()


# ---------------------------------------------------------------------------
# Sound synthesis helpers
# ---------------------------------------------------------------------------

def _envelope(length: int, attack: int = 0, decay: int | None = None) -> np.ndarray:
    """Create an amplitude envelope."""
    env = np.ones(length, dtype=np.float64)
    if attack > 0:
        env[:attack] = np.linspace(0, 1, attack)
    if decay is not None and decay > 0:
        start = max(0, length - decay)
        env[start:] = np.linspace(1, 0, length - start)
    return env


def _noise(length: int) -> np.ndarray:
    """White noise."""
    return np.random.default_rng(42).uniform(-1, 1, length)


def _sine(freq: float, length: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Sine wave."""
    t = np.arange(length) / sr
    return np.sin(2 * np.pi * freq * t)


def _square(freq: float, length: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Square wave."""
    return np.sign(_sine(freq, length, sr))


# ---------------------------------------------------------------------------
# Individual sound generators
# ---------------------------------------------------------------------------

def _synth_shot() -> np.ndarray:
    """Short sharp noise burst — puck shot."""
    n = int(SAMPLE_RATE * 0.06)
    samples = _noise(n) * _envelope(n, attack=0, decay=int(n * 0.8))
    return (samples * 20000).astype(np.float64)


def _synth_pass() -> np.ndarray:
    """Softer version of shot."""
    n = int(SAMPLE_RATE * 0.04)
    samples = _noise(n) * _envelope(n, attack=0, decay=int(n * 0.7))
    return (samples * 12000).astype(np.float64)


def _synth_wall_bounce() -> np.ndarray:
    """Brief thump — puck hitting boards."""
    n = int(SAMPLE_RATE * 0.05)
    thump = _sine(120, n) * _envelope(n, decay=int(n * 0.9))
    click = _noise(int(SAMPLE_RATE * 0.015)) * _envelope(int(SAMPLE_RATE * 0.015), decay=int(SAMPLE_RATE * 0.012))
    # Pad click to match length
    padded_click = np.zeros(n)
    padded_click[: len(click)] = click
    samples = thump * 0.6 + padded_click * 0.4
    return (samples * 18000).astype(np.float64)


def _synth_body_check() -> np.ndarray:
    """Heavier impact — body check."""
    n = int(SAMPLE_RATE * 0.1)
    low_thump = _sine(80, n) * _envelope(n, attack=int(n * 0.05), decay=int(n * 0.7))
    noise_burst = _noise(n) * _envelope(n, decay=int(n * 0.6))
    samples = low_thump * 0.5 + noise_burst * 0.5
    return (samples * 22000).astype(np.float64)


def _synth_pickup() -> np.ndarray:
    """Quick rising tone — puck pickup."""
    n = int(SAMPLE_RATE * 0.05)
    t = np.arange(n) / SAMPLE_RATE
    freq = 300 + 400 * (t / t[-1])  # sweep 300 -> 700 Hz
    samples = np.sin(2 * np.pi * freq * t) * _envelope(n, decay=int(n * 0.3))
    return (samples * 10000).astype(np.float64)


def _synth_goal_horn() -> np.ndarray:
    """Layered horn blast — GOAL!"""
    n = int(SAMPLE_RATE * 2.0)
    env = _envelope(n, attack=int(SAMPLE_RATE * 0.15), decay=int(SAMPLE_RATE * 0.4))
    horn = (
        _sine(220, n) * 0.4
        + _sine(330, n) * 0.3
        + _sine(440, n) * 0.2
        + _sine(550, n) * 0.1
    )
    samples = horn * env
    return (samples * 24000).astype(np.float64)


def _synth_whistle() -> np.ndarray:
    """Referee whistle — high sine with vibrato."""
    n = int(SAMPLE_RATE * 0.5)
    t = np.arange(n) / SAMPLE_RATE
    vibrato = 15 * np.sin(2 * np.pi * 8 * t)  # 8 Hz vibrato
    freq = 3000 + vibrato
    samples = np.sin(2 * np.pi * np.cumsum(freq) / SAMPLE_RATE)
    samples *= _envelope(n, attack=int(SAMPLE_RATE * 0.02), decay=int(SAMPLE_RATE * 0.1))
    return (samples * 14000).astype(np.float64)


def _synth_buzzer() -> np.ndarray:
    """Period-end buzzer — low square wave."""
    n = int(SAMPLE_RATE * 1.0)
    samples = _square(150, n) * _envelope(n, attack=int(SAMPLE_RATE * 0.01), decay=int(SAMPLE_RATE * 0.2))
    return (samples * 18000).astype(np.float64)
