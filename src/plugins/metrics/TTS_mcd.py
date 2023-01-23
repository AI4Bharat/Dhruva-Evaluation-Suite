import argparse
import fnmatch
import logging
import multiprocessing as mp
import os
from typing import Dict, List, Tuple
from plugins.metrics.config import IndicTTSConfig
from plugins import PluginBase
from tqdm import tqdm
import time
import librosa
import numpy as np
import pysptk
import soundfile as sf
from fastdtw import fastdtw
from scipy import spatial

class TTS_mcd(PluginBase):
    def __init__(self, **kwargs):
        self.config = IndicTTSConfig()
    def find_files(self,
        root_dir: str, query: List[str] = ["*.flac", "*.wav"], include_root_dir: bool = True
    ) -> List[str]:
        """Find files recursively.
        Args:
            root_dir (str): Root root_dir to find.
            query (List[str]): Query to find.
            include_root_dir (bool): If False, root_dir name is not included.
        Returns:
            List[str]: List of found filenames.
        """
        files = []
        for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
            for q in query:
                for filename in fnmatch.filter(filenames, q):
                    files.append(os.path.join(root, filename))
        if not include_root_dir:
            files = [file_.replace(root_dir + "/", "") for file_ in files]

        return files


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
                n_fft=self.config.n_fft,
                n_shift=self.config.n_shift,
                mcep_dim=self.config.mcep_dim,
                mcep_alpha=self.config.mcep_alpha,
            )
            gt_mcep = self.sptk_extract(
                x=gt_x,
                fs=fs,
                n_fft=self.config.n_fft,
                n_shift=self.config.n_shift,
                mcep_dim=self.config.mcep_dim,
                mcep_alpha=self.config.mcep_alpha,
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


    
    def invoke(self):
        """Run MCD calculation in parallel."""

        # find files
        if os.path.isdir(self.config.gen_wavdir_or_wavscp):
            gen_files = sorted(self.find_files(self.config.gen_wavdir_or_wavscp))
        else:
            with open(self.config.gen_wavdir_or_wavscp) as f:
                gen_files = [line.strip().split(None, 1)[1] for line in f.readlines()]
            if gen_files[0].endswith("|"):
                raise ValueError("Not supported wav.scp format.")
        if os.path.isdir(self.config.gt_wavdir_or_wavscp):
            gt_files = sorted(self.find_files(self.config.gt_wavdir_or_wavscp))
        else:
            with open(self.config.gt_wavdir_or_wavscp) as f:
                gt_files = [line.strip().split(None, 1)[1] for line in f.readlines()]
            if gt_files[0].endswith("|"):
                raise ValueError("Not supported wav.scp format.")

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
        file_lists = np.array_split(gen_files, self.config.nj)
        file_lists = [f_list.tolist() for f_list in file_lists]

        # multi processing
        with mp.Manager() as manager:
            mcd_dict = None
            while not mcd_dict:
                try:
                    mcd_dict = manager.dict()
                except:
                    pass
                print('error... will retry...')
                time.sleep(2)
            processes = []
            for f in file_lists:
                p = mp.Process(target=self.calculate, args=(f, gt_files, mcd_dict))
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

        # write results
        if self.config.outdir is None:
            if os.path.isdir(self.config.gen_wavdir_or_wavscp):
                self.config.outdir = self.config.gen_wavdir_or_wavscp
            else:
                self.config.outdir = os.path.dirname(self.config.gen_wavdir_or_wavscp)
        os.makedirs(self.config.outdir, exist_ok=True)
        with open(f"{self.config.outdir}/utt2mcd", "w") as f:
            for utt_id in sorted(mcd_dict.keys()):
                mcd = mcd_dict[utt_id]
                f.write(f"{utt_id} {mcd:.4f}\n")
        with open(f"{self.config.outdir}/mcd_avg_result.txt", "w") as f:
            f.write(f"#utterances: {len(gen_files)}\n")
            f.write(f"Average: {mean_mcd:.4f} ± {std_mcd:.4f}")

        logging.info("Successfully finished MCD evaluation.")

