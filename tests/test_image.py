"""Tests for image helpers."""

from types import SimpleNamespace

from vlm_agent_gateway.image import encode_image


def test_encode_image_url_uses_timeout_and_status_check(monkeypatch):
    """Remote image fetches should enforce timeout and raise on HTTP errors."""
    calls: dict[str, object] = {}

    def fake_resize_with_padding(image: bytes, target_size):
        calls["resized"] = (image, target_size)
        return b"jpeg-bytes"

    def fake_get(url: str, timeout: int):
        calls["request"] = (url, timeout)

        def raise_for_status():
            calls["status_checked"] = True

        return SimpleNamespace(content=b"raw-image", raise_for_status=raise_for_status)

    monkeypatch.setattr("vlm_agent_gateway.image.resize_with_padding", fake_resize_with_padding)
    monkeypatch.setattr("vlm_agent_gateway.image.requests.get", fake_get)

    encoded = encode_image("https://example.com/image.jpg", resize=True)

    assert encoded
    assert calls["request"] == ("https://example.com/image.jpg", 30)
    assert calls["status_checked"] is True
    assert calls["resized"] == (b"raw-image", (512, 512))
