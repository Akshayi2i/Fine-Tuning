"""scripts/check_env_file.py: the lines of a .env that bash would misread when it sources the file."""

from __future__ import annotations

from scripts.check_env_file import main, problems


def test_ordinary_lines_are_safe():
    text = "\n".join([
        "# a comment", "", "A=1", "export B=https://x.blob.core.windows.net/", "C=", "D=/workspace/.cache/hf",
        "E='a b;c&d'", 'F="sv=1&sig=x"', "G=plain   # a trailing comment",
    ])
    assert problems(text) == []


def test_an_unquoted_value_with_a_space_is_named():
    found = problems("NAME=Akshay Pimpale\n")
    assert len(found) == 1 and "line 1" in found[0] and "NAME" in found[0] and "a space" in found[0]


def test_an_unquoted_semicolon_or_ampersand_is_named():
    assert len(problems("A=a;b\nB=sv=1&sig=x\n")) == 2


def test_a_line_that_is_not_key_value_is_named():
    assert "run it as a command" in problems("just some words\n")[0]


def test_a_file_saved_on_windows_is_read_like_any_other():
    assert problems("A=1\r\nB=2\r\n") == []
    assert len(problems("A=1 2\r\n")) == 1


def test_a_double_quoted_value_with_a_dollar_is_named():
    assert len(problems('A="pa$word"\n')) == 1


def test_the_message_never_holds_the_value():
    assert "hunter2" not in " ".join(problems("TOKEN=hunter2 extra\n"))


def test_the_command_line_exit_code(tmp_path, capsys):
    good, bad = tmp_path / "good.env", tmp_path / "bad.env"
    good.write_text("A=1\n", encoding="utf-8")
    bad.write_text("A=1 2\n", encoding="utf-8")
    assert main([str(good)]) == 0
    assert main([str(bad)]) == 1 and "line 1" in capsys.readouterr().err
    assert main([str(tmp_path / "missing.env")]) == 0
