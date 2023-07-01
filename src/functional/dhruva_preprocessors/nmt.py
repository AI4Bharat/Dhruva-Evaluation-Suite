from constants import FLORES_TO_ULCA_LANGUAGE_CODE_MAPPING


def normalize_language_codes(raw_data, source_language, target_language):
    raw_data["source_language"] = FLORES_TO_ULCA_LANGUAGE_CODE_MAPPING.get(source_language)
    raw_data["target_language"] = FLORES_TO_ULCA_LANGUAGE_CODE_MAPPING.get(target_language)
    return raw_data
