# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import pytest
from src.datasets_lib.responses_io import ResponsesLoader


@pytest.fixture(scope="module")
def responses_loader():
    loader = ResponsesLoader(
        "./tests/data/",
        from_folders=["toxicity-responses/tiny-gpt2/jigsaw/*/transformer.*"],
        label_map={"toxic": 1, "non-toxic": 0},
    )
    return loader


def test_get_attribute_values(responses_loader):
    attribute_name = "module_names"
    values = responses_loader.get_attribute_values(attribute_name)

    # Check if the returned set is not empty
    assert len(values) > 0, f"No values found for attribute {attribute_name}"


def test_load_data_subset(responses_loader):
    filter = {"pooling_op": "max"}
    data = responses_loader.load_data_subset(filter)

    # Check if the returned dictionary is not empty
    assert len(data) > 0, "No data loaded"

    filter = {"module_names": "transformer.h.0.mlp.c_proj"}
    data = responses_loader.load_data_subset(filter)
    assert set(data["subset"]) == set(["toxic", "non-toxic"])
    assert set(data["module_names"]) == set(["transformer.h.0.mlp.c_proj"])

    filter = {"module_names": "transformer.h.0.mlp.c_proj", "subset": "non-toxic"}
    data = responses_loader.load_data_subset(filter)
    assert set(data["subset"]) == set(["non-toxic"])
    assert set(data["module_names"]) == set(["transformer.h.0.mlp.c_proj"])

    fail = False
    try:
        responses_loader.load_data_subset({"APPLE": "PIE"})
    except:
        fail = True
    assert fail, "load_data_subset should fail with unknown keys."
