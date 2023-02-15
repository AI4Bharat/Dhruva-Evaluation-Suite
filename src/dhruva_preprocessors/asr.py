from .speech_transcript_cleaning import cleaning_pipeline, get_dict_chars


def clean_and_normalize_transcripts(raw_data):
    dict_characters = get_dict_chars(raw_data["language"])
    no_ood, preprocessed_sentence = cleaning_pipeline(
        dict_characters, raw_data["transcript"], raw_data["language"]
    )
    return {"audio": raw_data["audio"], "transcript": preprocessed_sentence, "language": raw_data["language"]}
