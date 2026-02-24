import torch
from copy import deepcopy
from cheetah.accelerator import Segment
from cheetah.particles import Beam, ParticleBeam


class CheetahSimulator:
    """
    Simulator class for Cheetah accelerator simulations.

    This class provides an interface to simulate the behavior
    of a particle beam as it travels through a Cheetah accelerator segment.
    It allows for tracking the beam, retrieving energy profiles,
    and controlling the beam shutter state.

    Attributes
    ----------
    segment : Segment
        The Cheetah Segment representing the accelerator configuration.
    initial_beam_distribution : Beam
        The initial beam distribution to be tracked through the segment.

    Methods
    -------
    reset()
        Resets the simulator to its initial state.
    track()
        Tracks the beam through the segment, updating the internal state.
    get_energy()
        Retrieves the energy of the beam at every element in the segment.
    set_shutter(value: bool)
        Sets the beam shutter state, controlling whether the beam is present or not.


    """

    def __init__(
        self,
        segment: Segment,
        initial_beam_distribution: Beam,
    ) -> None:
        """
        Simulator class for Cheetah accelerator simulations.

        Parameters
        ----------
        segment : Segment
            The Cheetah Segment representing the accelerator configuration.
        initial_beam_distribution : Beam
            The initial beam distribution to be tracked through the segment.
        shutter_pv : str, optional
            The process variable name for the beam shutter, if applicable.

        """

        self.segment = segment
        self._initial_segment = deepcopy(segment)
        self.initial_beam_distribution = initial_beam_distribution.clone()
        self.beam_distribution = self.initial_beam_distribution.clone()
        self.initial_beam_distribution_charge = (
            initial_beam_distribution.particle_charges
        )

        self.track()
        self.energies = self.get_energy()

    def reset(self):
        self.segment = deepcopy(self._initial_segment)
        self.beam_distribution = self.initial_beam_distribution.clone()
        self.track()
        self.energies = self.get_energy()

    def track(self):
        self.segment.track(self.beam_distribution)

    def get_energy(self):
        """
        Get the energy of the beam in the virtual accelerator simulator at
        every element for use in calculating the magnetic rigidity.

        Note: need to track on a copy of the segment to not influence readings!
        """
        test_beam = ParticleBeam(
            torch.zeros(1, 7), energy=self.beam_distribution.energy
        )
        test_segment = deepcopy(self.segment)
        element_names = [e.name for e in test_segment.elements]
        return dict(
            zip(
                element_names,
                test_segment.get_beam_attrs_along_segment(("energy",), test_beam)[0],
            )
        )

    def set_shutter(self, value: bool):
        """
        Set the beam shutter state in the virtual accelerator simulator.
        If `value` is True, the shutter is closed (no beam), otherwise it is open (beam present).
        """
        if value:
            self.beam_distribution.particle_charges = torch.tensor(0.0)
        else:
            self.beam_distribution.particle_charges = (
                self.initial_beam_distribution_charge.clone()
            )
