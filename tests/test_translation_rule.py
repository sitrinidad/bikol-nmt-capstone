"""Rule: every sentence goes to the model — there is no "Dai ko isi" / "Mai ko isi" / "I don't know." fallback
for sentences that are not in the dataset."""
from pathlib import Path

import pytest

from backend.nmt import EngineState, NMTEngine


class FakeModelEngine(NMTEngine):
    """The neural model is replaced by an echo so no 2.5 GB download is needed."""

    def __init__(self):
        super().__init__(Path("unused"))
        self.state = EngineState.READY
        self.model_inputs = []

    def _generate(self, segments, source_lang, target_lang):
        self.model_inputs.extend(segments)
        return [f"MODEL({s})" for s in segments], [-0.1] * len(segments), [5] * len(segments)


@pytest.mark.parametrize("source,target", [("eng", "bcl_South"), ("eng", "bcl_North"), ("bcl_South", "eng")])
def test_sentence_not_in_the_dataset_is_translated_by_the_model(source, target):
    engine = FakeModelEngine()
    result = engine.translate("I love pizza.", source, target)
    assert result.translated_text == "MODEL(I love pizza.)"
    assert result.translated_text not in ("Dai ko isi", "Mai ko isi", "I don't know.")
    assert result.confidence_score is not None
    assert engine.model_inputs == ["I love pizza."]


def test_every_sentence_of_a_mixed_input_goes_to_the_model():
    engine = FakeModelEngine()
    result = engine.translate("Look at this. I love pizza.", "eng", "bcl_South")
    assert engine.model_inputs == ["Look at this.", "I love pizza."]
    assert result.translated_text == "MODEL(Look at this.) MODEL(I love pizza.)"
