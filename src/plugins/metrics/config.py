import os
from typing import List
from pathlib import Path
from dataclasses import dataclass


@dataclass
class BaseConfig():
    gen_wavdir_or_wavscp : str
    gt_wavdir_or_wavscp : str
    outdir : str

@dataclass
class IndicTTSConfig(BaseConfig):
    #Path of directory or wav.scp for generated waveforms.
    gen_wavdir_or_wavscp : str = "../datasets/outputs/IndicTTS/Hindi/test/"
    #Path of directory or wav.scp for ground truth waveforms.
    gt_wavdir_or_wavscp : str =  "../datasets/raw/IndicTTS/Hindi/wavs-22k/"
    #Path of directory to write the results.
    outdir : str = "../datasets/outputs/IndicTTS/Hindi/f0"
    # Dimension of mel cepstrum coefficients. If None, automatically set to the best dimension for the sampling.
    mcep_dim :int = None
    # All pass constant for mel-cepstrum analysis. If None, automatically set to the best dimension for the sampling.
    mcep_alpha : float = None
    # The number of FFT points
    n_fft : int = 1024
    # The number of shift points
    n_shift : int = 256
    # Minimum f0 value.
    f0min : int = 40
    # Maximum f0 value.
    f0max : int = 800
    # Number of parallel jobs.
    nj : int = 16
    # Verbosity level. Higher is more logging.
    verbose : int = 1

    def __post_init__(self):
        Path(os.path.dirname(self.outdir)).mkdir(parents=True, exist_ok=True)