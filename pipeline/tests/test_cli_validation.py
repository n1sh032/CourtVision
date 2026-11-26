import pytest

from courtvision.cli import main


def test_existing_output_is_preserved(tmp_path):
    output = tmp_path / "input.mp4"
    output.write_bytes(b"original")
    with pytest.raises(SystemExit) as error:
        main(["overlay", str(output), "pose.json", "-o", str(output)])
    assert error.value.code == 2
    assert output.read_bytes() == b"original"
