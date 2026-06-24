"""Tests for md_to_tg_html converter."""

from __future__ import annotations

from src.utils.text import md_to_tg_html


def test_bold_italic_code_converted() -> None:
    """md_to_tg_html converts markdown formatting to Telegram HTML."""
    # Bold
    assert md_to_tg_html("**bold**") == "<b>bold</b>"

    # Italic
    assert md_to_tg_html("*italic*") == "<i>italic</i>"

    # Inline code
    assert md_to_tg_html("`code`") == "<code>code</code>"

    # Code blocks
    assert md_to_tg_html("```\nблоки\n```") == "<pre>блоки</pre>"

    # Strikethrough
    assert md_to_tg_html("~~strike~~") == "<s>strike</s>"

    # HTML special chars escaped
    assert md_to_tg_html("a < b & c > d") == "a &lt; b &amp; c &gt; d"

    # Nested: bold with italic inside
    assert md_to_tg_html("**bold *and italic***") == "<b>bold <i>and italic</i></b>"

    # Plaintext unchanged
    assert md_to_tg_html("hello world") == "hello world"

    # Mixed content
    result = md_to_tg_html("Hello **world** and *foo*")
    assert result == "Hello <b>world</b> and <i>foo</i>"

    # HTML chars inside formatting
    assert md_to_tg_html("**<script>**") == "<b>&lt;script&gt;</b>"

    # Code block with language
    assert md_to_tg_html("```python\nprint('hi')\n```") == "<pre>print('hi')</pre>"
