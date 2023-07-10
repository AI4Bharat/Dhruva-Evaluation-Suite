from .speech_transcript_cleaning import cleaning_pipeline, get_dict_chars


def clean_and_normalize_transcripts(raw_data, label_column, source_language):
    dict_characters = get_dict_chars(source_language)
    no_ood, preprocessed_sentence = cleaning_pipeline(dict_characters, raw_data[label_column], source_language)
    raw_data[label_column] = preprocessed_sentence
    return raw_data
