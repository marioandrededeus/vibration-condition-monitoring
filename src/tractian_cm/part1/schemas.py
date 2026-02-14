from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator


class CarpetRegion(BaseModel):
    """
    EN: Output contract for Part 1.
    EN: Regions must be explicit frequency intervals to support reproducible segmentation.
    """
    start_hz: float = Field(..., ge=0.0, description="Start frequency in Hz")
    end_hz: float = Field(..., ge=0.0, description="End frequency in Hz")

    @field_validator("end_hz")
    @classmethod
    def end_must_be_greater_than_start(cls, v: float, info):
        # EN: Ensures region boundaries are valid (non-negative, non-empty).
        start = info.data.get("start_hz")
        if start is not None and v <= start:
            raise ValueError("end_hz must be greater than start_hz")
        return v


class Wave(BaseModel):
    """
    EN: Input waveform contract.
    EN: For vibration analysis, the frequency axis depends on a correct time axis.
    """
    time: List[float] = Field(..., min_length=2, description="Time points in seconds")
    signal: List[float] = Field(..., min_length=2, description="Signal values (e.g., acceleration in g)")

    @field_validator("signal")
    @classmethod
    def signal_same_length_as_time(cls, v: List[float], info):
        # EN: Time-series operations (PSD/FFT) require aligned time and signal arrays.
        t = info.data.get("time")
        if t is not None and len(v) != len(t):
            raise ValueError("signal must have the same length as time")
        return v
