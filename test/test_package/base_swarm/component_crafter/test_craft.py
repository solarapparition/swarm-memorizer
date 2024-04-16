"""Test the `craft` module."""

from base_swarm.bots.component_crafter.craft import specify_function


def test_specify_function():
    """Test the `specify_function` function."""
    specify_function(context="dummy", requirements="dummy")