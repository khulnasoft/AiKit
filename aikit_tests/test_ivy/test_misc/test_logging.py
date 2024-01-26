import pytest
import logging
import aikit


def test_invalid_logging_mode():
    with pytest.raises(AssertionError):
        aikit.set_logging_mode("INVALID")


def test_set_logging_mode():
    aikit.set_logging_mode("DEBUG")
    assert logging.getLogger().level == logging.DEBUG

    aikit.set_logging_mode("INFO")
    assert logging.getLogger().level == logging.INFO

    aikit.set_logging_mode("WARNING")
    assert logging.getLogger().level == logging.WARNING

    aikit.set_logging_mode("ERROR")
    assert logging.getLogger().level == logging.ERROR


def test_unset_logging_mode():
    aikit.set_logging_mode("DEBUG")
    aikit.set_logging_mode("INFO")
    aikit.unset_logging_mode()
    assert logging.getLogger().level == logging.DEBUG
