from dataclasses import dataclass
from typing import Any
from impact import Impact
from lume import ReadOnlyActionMixin
import numpy as np
from lume.variables import NDVariable, ScalarVariable, IntVariable


@dataclass(frozen=True)
class ScreenSpec:
    """Definition of a screen detector geometry in the lattice."""

    element_name: str
    shape: tuple[int, int]
    pixel_size: float

    @property
    def half_width(self) -> tuple[float, float]:
        """Half-width used to build the histogram range."""
        return (
            (self.shape[0] * self.pixel_size) / 2,
            (self.shape[1] * self.pixel_size) / 2,
        )


class _ScreenSpecVariableMixin:
    """Shared ScreenSpec conversion utilities for screen-derived variables."""

    element_name: str
    pixel_size: float
    shape: tuple[int, int]

    @property
    def screen_spec(self) -> ScreenSpec:
        """Canonical screen specification for this variable."""
        return ScreenSpec(
            element_name=self.element_name,
            shape=self.shape,
            pixel_size=self.pixel_size,
        )

    @classmethod
    def _from_screen_spec(
        cls,
        *,
        name: str,
        screen_spec: ScreenSpec,
        **kwargs,
    ):
        """Construct a screen-derived variable from a shared ScreenSpec."""
        return cls(
            name=name,
            element_name=screen_spec.element_name,
            pixel_size=screen_spec.pixel_size,
            shape=screen_spec.shape,
            **kwargs,
        )


class ScreenImageVariable(_ScreenSpecVariableMixin, NDVariable, ReadOnlyActionMixin):
    """Read-only action that returns a screen image at a given element. The image is normalized to a unit scale"""

    pixel_size: float  # default pixel size in meters
    read_only: bool = True

    @classmethod
    def from_screen_spec(cls, name: str, screen_spec: ScreenSpec, **kwargs):
        """Build a screen-image variable from a shared screen specification."""
        return cls._from_screen_spec(
            name=name,
            screen_spec=screen_spec,
            **kwargs,
        )

    def _get(self, simulator: Impact) -> Any:

        beam = simulator.particles[self.element_name]

        # simple screen that counts number of particles
        half_width = self.screen_spec.half_width
        hist, _ = beam.histogramdd(
            "x",
            "y",
            bins=self.shape,
            range=((-half_width[0], half_width[0]), (-half_width[1], half_width[1])),
        )

        # normalize to unit scale
        hist /= np.max(hist) if np.max(hist) > 0 else 1.0

        return hist


class ScreenResolutionVariable(
    _ScreenSpecVariableMixin, ScalarVariable, ReadOnlyActionMixin
):
    """Read-only action that returns the pixel size in microns."""

    pixel_size: float
    unit: str = "um"  # default unit is microns
    read_only: bool = True

    @classmethod
    def from_screen_spec(cls, name: str, screen_spec: ScreenSpec, **kwargs):
        """Build a screen-resolution variable from a shared screen specification."""
        return cls._from_screen_spec(
            name=name,
            screen_spec=screen_spec,
            **kwargs,
        )

    def _get(self, simulator: Impact) -> Any:
        _ = simulator
        return self.screen_spec.pixel_size


class ScreenImageShapeVariable(
    _ScreenSpecVariableMixin, IntVariable, ReadOnlyActionMixin
):
    """Read-only action that returns the pixel shape of the screen image."""

    index: int  # index of the dimension to return (0 for x, 1 for y)
    read_only: bool = True

    @classmethod
    def from_screen_spec(cls, name: str, screen_spec: ScreenSpec, **kwargs):
        """Build a screen-image-size variable from a shared screen specification."""
        return cls._from_screen_spec(
            name=name,
            screen_spec=screen_spec,
            **kwargs,
        )

    def _get(self, simulator: Impact) -> Any:
        _ = simulator
        return self.screen_spec.shape[self.index]
