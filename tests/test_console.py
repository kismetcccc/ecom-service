import io

from app.console import safe_print


def test_safe_print_does_not_raise_on_ascii_only_stream():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="ascii", errors="strict")

    safe_print("💭 [思考] 你好", file=stream, flush=True)

    output = raw.getvalue().replace(b"\r\n", b"\n")
    assert output == b"\\U0001f4ad [\\u601d\\u8003] \\u4f60\\u597d\n"
