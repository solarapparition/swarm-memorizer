"""Test the `craft` module."""

from swarms.core_bots.component_crafter.craft import generate_test_writing_reasoning


def test_specify_function():
    """Test the `specify_function` function."""
    generate_test_writing_reasoning(context="dummy", requirements=["dummy"])