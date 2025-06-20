import pytest

from splitter_mr.model.base_model import BaseModel


# 1. Test instantiating BaseModel raises TypeError
def test_basemodel_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseModel()


# 2. Dummy subclass for testing
class DummyModel(BaseModel):
    def get_client(self):
        return "dummy-client"

    def extract_text(self, file: str, prompt: str, **parameters) -> str:
        # just echo the input for testing
        return f"extract:{prompt}"

    def analyze_resource(
        self, file: str, context: str, prompt: str, **parameters
    ) -> str:
        # just echo the inputs for testing
        return f"analyze:{prompt}|{context}"


def test_dummy_model_instantiable_and_methods_work():
    dummy = DummyModel()
    assert dummy.get_client() == "dummy-client"
    assert dummy.extract_text("b64", "PROMPT") == "extract:PROMPT"
    assert (
        dummy.analyze_resource("b64", "CONTEXT", "PROMPT") == "analyze:PROMPT|CONTEXT"
    )
