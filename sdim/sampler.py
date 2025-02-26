from typing import Optional, Tuple, Union
import numpy as np

class CompiledMeasurementSampler():
    def __init__(self,
                 circuit: object,
                 *,
                 skip_reference_sample: bool = False,
                 seed: Optional[int] = None,
                 reference_sample: np.ndarray = None,
    ) -> None:
        self.circuit = circuit
        self.skip_reference_sample = skip_reference_sample
        self.seed = seed
        self.reference_sample = reference_sample

    def sample(
            self,
            shots: int,
    ) -> np.ndarray:
        """
        Samples the measurement results of the circuit.

        Args:
            shots (int): Number of shots.

        Returns:
            np.ndarray: Measurement results.
        """
        if self.skip_reference_sample:
            return self._sample(shots)
        else:
            return self._sample_with_reference(shots)
    
    def sample_write(
            self,
            shots: int,
            filepath: str,
            format: str = '01',
    ) -> None:
        """
        Samples the measurement results of the circuit and writes them to a file.

        Args:
            shots (int): Number of shots.
            filename (str): File name.
        """
        ...
    

class CompiledDetectorSampler():
    def __init__(self,
                 circuit: object,
                 *,
                 seed: Optional[int] = None,
    ) -> None:
        ...
    def sample(
            self,
            shots: int,
            *,
            dets_out: Optional[np.ndarray] = None,
            obs_out: Optional[np.ndarray] = None,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        ...
    def sample_write( # TODO: Fix arguments to match Stim
            self,
            shots: int,
            filepath: str,
            format: str = '01',
    ) -> None:
        ...