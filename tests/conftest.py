"""Functional tests only. Anything that needs the OP-1 field skips cleanly when it is not connected."""
import os
import pytest

from op_bridge.device import find_midi_output, find_audio_input


def field_present() -> bool:
    return find_midi_output() is not None and find_audio_input() is not None


needs_field = pytest.mark.skipif(not field_present(), reason="OP-1 field not connected over USB in normal mode")


@pytest.fixture(scope="session")
def home(tmp_path_factory):
    """Point OP_BRIDGE_HOME at a temp dir so tests never touch the real config or sessions."""
    d = tmp_path_factory.mktemp("op-bridge-home")
    os.environ["OP_BRIDGE_HOME"] = str(d)
    import importlib
    from op_bridge import session
    importlib.reload(session)
    return str(d)
