import os
import json
# from pathlib import Path
from typing import List, Any
from collections import defaultdict

from tqdm import tqdm
from joblib import Parallel, delayed

INDIC_NLP_LIB_HOME = "indic_nlp_library"
INDIC_NLP_RESOURCES = "indic_nlp_resources"

from sacrebleu.metrics import BLEU
# from indicnlp import common
# common.set_resources_path(INDIC_NLP_RESOURCES)
# from indicnlp import loader
# loader.load()

from sacremoses import MosesTokenizer
from sacremoses import MosesDetokenizer
from sacremoses import MosesPunctNormalizer

from indicnlp.tokenize import indic_tokenize
from indicnlp.tokenize import indic_detokenize
from indicnlp.normalize import indic_normalize
from indicnlp.transliterate import unicode_transliterate

from plugins import PluginBase
# from plugins.scorers.config import ()


en_tok = MosesTokenizer(lang="en")
en_normalizer = MosesPunctNormalizer()


class BLEUMetric(PluginBase):
    """
    BLEU scores
    """

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def preprocess_line(self, line, normalizer, lang, transliterate=False):
        if lang == "en":
            return " ".join(
                en_tok.tokenize(en_normalizer.normalize(line.strip()), escape=False)
            )
        elif transliterate:
            # line = indic_detokenize.trivial_detokenize(line.strip(), lang)
            return unicode_transliterate.UnicodeIndicTransliterator.transliterate(
                " ".join(
                    indic_tokenize.trivial_tokenize(
                        normalizer.normalize(line.strip()), lang
                    )
                ),
                lang,
                "hi",
            ).replace(" ् ", "्")
        else:
            # we only need to transliterate for joint training
            return " ".join(
                indic_tokenize.trivial_tokenize(normalizer.normalize(line.strip()), lang)
            )

    def invoke(self, *args, **kwargs) -> Any:
        if not os.path.exists(self.kwargs["scorer_output"]):
            self._logger.error("No output file! SKipping metric calculation...")
            return False

        bleu = BLEU()
        self._logger.info("Processing output file...")

        hypothesis = []
        ground_truth = []
        with open(self.kwargs["scorer_output"], "r") as ipf:
            prev_normalizer = None
            prev_target_lang = None
            for line in tqdm(ipf):
                data = json.loads(line)

                if data["target_language"] != "en":
                    if not prev_normalizer or data["target_language"] != prev_target_lang:
                        # loop over all and get target languages
                        # normalise to target language if it is indic
                        normfactory = indic_normalize.IndicNormalizerFactory()
                        prev_normalizer = normfactory.get_normalizer(data["target_language"])
                        prev_target_lang = data["target_language"]

                    data["output"] = self.preprocess_line(data["output"], prev_normalizer, target_lang, transliterate=False)
                hypothesis.append(data["output"])
                ground_truth.append(data["target_sentence"])

        self._logger.info("Calculating BLEU...")
        score = bleu.corpus_score(hypothesis, ground_truth)
        self._logger.info(f"BLEU is: {score}")
