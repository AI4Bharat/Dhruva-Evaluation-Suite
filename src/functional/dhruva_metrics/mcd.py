#!/usr/bin/env python3

# Copyright 2020 Wen-Chin Huang and Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate MCD between generated and groundtruth audios with SPTK-based mcep."""

import argparse
import fnmatch
import logging
import multiprocessing as mp
import os
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pysptk
import soundfile as sf
from fastdtw import fastdtw
from scipy import spatial
import evaluate
import datasets
_KWARGS_DESCRIPTION = """
"""

_CITATION = """
"""

_DESCRIPTION = """
"""

class MCD(evaluate.Metric):
    
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("string")),
                    "references": datasets.Sequence(datasets.Value("string")),
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("string"),
                    "references": datasets.Value("string"),
                }
            ),
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html"],
        )


    def sptk_extract(self,
        x: np.ndarray,
        fs: int,
        n_fft: int = 512,
        n_shift: int = 256,
        mcep_dim: int = 25,
        mcep_alpha: float = 0.41,
        is_padding: bool = False,
    ) -> np.ndarray:
        """Extract SPTK-based mel-cepstrum.

        Args:
            x (ndarray): 1D waveform array.
            fs (int): Sampling rate
            n_fft (int): FFT length in point (default=512).
            n_shift (int): Shift length in point (default=256).
            mcep_dim (int): Dimension of mel-cepstrum (default=25).
            mcep_alpha (float): All pass filter coefficient (default=0.41).
            is_padding (bool): Whether to pad the end of signal (default=False).

        Returns:
            ndarray: Mel-cepstrum with the size (N, n_fft).

        """
        # perform padding
        if is_padding:
            n_pad = n_fft - (len(x) - n_fft) % n_shift
            x = np.pad(x, (0, n_pad), "reflect")

        # get number of frames
        n_frame = (len(x) - n_fft) // n_shift + 1

        # get window function
        win = pysptk.sptk.hamming(n_fft)

        # check mcep and alpha
        if mcep_dim is None or mcep_alpha is None:
            mcep_dim, mcep_alpha = self._get_best_mcep_params(fs)

        # calculate spectrogram
        mcep = [
            pysptk.mcep(
                x[n_shift * i : n_shift * i + n_fft] * win,
                mcep_dim,
                mcep_alpha,
                eps=1e-6,
                etype=1,
            )
            for i in range(n_frame)
        ]

        return np.stack(mcep)


    def _get_basename(self,path: str) -> str:
        return os.path.splitext(os.path.split(path)[-1])[0]


    def _get_best_mcep_params(self,fs: int) -> Tuple[int, float]:
        if fs == 16000:
            return 23, 0.42
        elif fs == 22050:
            return 34, 0.45
        elif fs == 24000:
            return 34, 0.46
        elif fs == 44100:
            return 39, 0.53
        elif fs == 48000:
            return 39, 0.55
        else:
            raise ValueError(f"Not found the setting for {fs}.")


    def calculate(self, 
        file_list: List[str],
        gt_file_list: List[str],
        mcd_dict: Dict,
    ):
        """Calculate MCD."""
        for i, gen_path in enumerate(file_list):
            corresponding_list = list(
                filter(lambda gt_path: self._get_basename(gt_path) in gen_path, gt_file_list)
            )
            assert len(corresponding_list) == 1
            gt_path = corresponding_list[0]
            gt_basename = self._get_basename(gt_path)

            # load wav file as int16
            gen_x, gen_fs = sf.read(gen_path, dtype="int16")
            gt_x, gt_fs = sf.read(gt_path, dtype="int16")

            fs = gen_fs
            if gen_fs != gt_fs:
                gt_x = librosa.resample(gt_x.astype(np.float), gt_fs, gen_fs)

            # extract ground truth and converted features
            gen_mcep = self.sptk_extract(
                x=gen_x,
                fs=fs,
                n_fft=1024,
                n_shift=256,
                mcep_dim=None,
                mcep_alpha=None,
            )
            gt_mcep = self.sptk_extract(
                x=gt_x,
                fs=fs,
                n_fft=1024,
                n_shift=256,
                mcep_dim=None,
                mcep_alpha=None,
            )

            # DTW
            _, path = fastdtw(gen_mcep, gt_mcep, dist=spatial.distance.euclidean)
            twf = np.array(path).T
            gen_mcep_dtw = gen_mcep[twf[0]]
            gt_mcep_dtw = gt_mcep[twf[1]]

            # MCD
            diff2sum = np.sum((gen_mcep_dtw - gt_mcep_dtw) ** 2, 1)
            mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)
            logging.info(f"{gt_basename} {mcd:.4f}")
            mcd_dict[gt_basename] = mcd



    def _compute(self, predictions, references):
        """Run MCD calculation in parallel."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        # find files
        gen_files = sorted(predictions)

        gt_files = sorted(references)

        # Get and divide list
        if len(gen_files) == 0:
            raise FileNotFoundError("Not found any generated audio files.")
        if len(gen_files) > len(gt_files):
            raise ValueError(
                "#groundtruth files are less than #generated files "
                f"(#gen={len(gen_files)} vs. #gt={len(gt_files)}). "
                "Please check the groundtruth directory."
            )
        logging.info("The number of utterances = %d" % len(gen_files))
        file_lists = np.array_split(gen_files, 16)
        file_lists = [f_list.tolist() for f_list in file_lists]

        # multi processing
        with mp.Manager() as manager:
            mcd_dict = manager.dict()
            processes = []
            for f in file_lists:
                p = mp.Process(target= self.calculate, args=(f, gt_files, mcd_dict))
                p.start()
                processes.append(p)

            # wait for all process
            for p in processes:
                p.join()

            # convert to standard list
            mcd_dict = dict(mcd_dict)

            # calculate statistics
            mean_mcd = np.mean(np.array([v for v in mcd_dict.values()]))
            std_mcd = np.std(np.array([v for v in mcd_dict.values()]))
            logging.info(f"Average: {mean_mcd:.4f} ± {std_mcd:.4f}")

        return {"MCD" : mean_mcd}

