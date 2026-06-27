import rs_words


def test_version():
    assert rs_words.__version__ == "0.1.0"


def test_config_paths():
    from rs_words import config
    assert config.ROOT.exists()
    assert config.DATA_DIR == config.ROOT / "data"
