from backend.nmt.preprocess import normalize_source, split_segments


def test_bikol_input_is_cleaned_like_phase1_training_data():
    assert normalize_source("  Marhay   na  aldâw !", "bcl_South") == "Maray na aldaw!"


def test_english_keeps_its_words():
    assert normalize_source("“Hello”  there", "eng") == '"Hello" there'


def test_text_is_split_into_lines_and_sentences():
    assert split_segments("Hello. How are you?\n\nGood!") == [["Hello.", "How are you?"], ["Good!"]]
