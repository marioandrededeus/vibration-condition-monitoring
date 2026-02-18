from typing import List

from tractian_cm.part1.schemas import Wave, CarpetRegion
from tractian_cm.part1.carpet_model import detect_carpet_regions
from tractian_cm.part1.config import CarpetConfig


class Model:
    """
    Carpet detection model wrapper.

    This class follows the interface specified in the case description:

        class Model:
            def predict(self, wave: Wave) -> List[CarpetRegion]

    It acts as a thin wrapper over the core detection engine
    implemented in detect_carpet_regions.
    """

    def __init__(self, config: CarpetConfig | None = None):
        """
        Initialize model with optional configuration.
        If no configuration is provided, default CarpetConfig is used.
        """
        self.config = config or CarpetConfig()

    def _infer_sampling_rate(self, wave: Wave) -> float:
        """
        Infer sampling frequency from time vector.
        Assumes uniform sampling.
        """
        if len(wave.t) < 2:
            raise ValueError("Wave must contain at least two time samples.")

        dt = wave.t[1] - wave.t[0]
        if dt <= 0:
            raise ValueError("Invalid time vector: non-positive time difference detected.")

        return 1.0 / dt

    def predict(self, wave: Wave) -> List[CarpetRegion]:
        """
        Detect carpet regions from a validated Wave object.

        Parameters
        ----------
        wave : Wave
            Validated wave object containing time vector and signal data.

        Returns
        -------
        List[CarpetRegion]
            List of detected carpet regions.
        """
        fs = self._infer_sampling_rate(wave)

        regions = detect_carpet_regions(
            wave=wave,
            fs=fs,
            cfg=self.config
        )

        return regions
