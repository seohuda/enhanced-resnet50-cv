import os

from train import setup_logging


def test_setup_logging_creates_file_and_writes_message(tmp_path):
    logger = setup_logging(str(tmp_path))
    logger.info("hello world")
    for handler in logger.handlers:
        handler.flush()

    log_path = os.path.join(str(tmp_path), "train.log")
    assert os.path.isfile(log_path)
    with open(log_path) as f:
        content = f.read()
    assert "hello world" in content


def test_setup_logging_does_not_duplicate_handlers_on_repeat_calls(tmp_path):
    logger1 = setup_logging(str(tmp_path))
    handler_count_after_first = len(logger1.handlers)

    logger2 = setup_logging(str(tmp_path))
    handler_count_after_second = len(logger2.handlers)

    assert handler_count_after_first == handler_count_after_second
