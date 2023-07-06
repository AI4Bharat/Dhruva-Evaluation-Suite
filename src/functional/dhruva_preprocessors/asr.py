from .speech_transcript_cleaning import cleaning_pipeline, get_dict_chars


def clean_and_normalize_transcripts(raw_data, label_column):
    language_column = "language"
    if "locale" in raw_data:
        language_column = "locale"
    dict_characters = get_dict_chars(raw_data[language_column])
    no_ood, preprocessed_sentence = cleaning_pipeline(
        dict_characters, raw_data[label_column], raw_data[language_column]
    )
    raw_data[label_column] = preprocessed_sentence
    return raw_data
