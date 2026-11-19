"""Files their owner alone may read or write (masw.io.private), whatever the system: POSIX's mode
0o600, or on Windows an access list granting the current user alone."""

from pathlib import Path

from masw.io.private import is_private, make_private, write_private


def test_a_private_file_is_written_whole_and_its_owners_alone(tmp_path: Path) -> None:
    path = tmp_path / "secret.env"
    write_private(path, "KEY=first\n")
    write_private(path, "KEY=second, é\n")

    assert path.read_text(encoding="utf-8") == "KEY=second, é\n"
    assert is_private(path)
    assert sorted(tmp_path.iterdir()) == [path]  # nothing left beside it


def test_a_file_others_may_read_is_made_its_owners_alone(tmp_path: Path) -> None:
    path = tmp_path / "copied.env"
    path.write_text("KEY=value\n", encoding="utf-8")
    path.chmod(0o644)  # on Windows, its folder's rights, inherited
    assert not is_private(path)

    make_private(path)

    assert is_private(path)
    assert path.read_text(encoding="utf-8") == "KEY=value\n"
