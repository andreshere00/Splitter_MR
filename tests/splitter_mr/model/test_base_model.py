import pytest

from splitter_mr.model.base_model import BaseModel

# Helpers


class DummyModel(BaseModel):
    def __init__(self, model_name="dummy-model"):
        self.model_name = model_name

    def get_client(self):
        return "dummy-client"

    def extract_text(self, prompt: str, file: bytes = None, **parameters) -> str:
        # just echo the input for testing
        return f"extract:{prompt}"


# Test cases
def test_basemodel_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseModel("foo")


def test_dummy_model_instantiable_and_methods_work():
    dummy = DummyModel()
    assert dummy.get_client() == "dummy-client"
    assert dummy.extract_text("PROMPT") == "extract:PROMPT"
