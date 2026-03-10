"""
Image processing utilities.
"""

import base64
import io

import requests
from PIL import Image, ImageOps


def is_url(path: str) -> bool:
    """Check if a string is an HTTP/HTTPS URL."""
    return path.startswith("http://") or path.startswith("https://")


def resize_with_padding(
    image: str | bytes,
    target_size: tuple[int, int] = (512, 512),
) -> bytes:
    """
    Resize an image to target size while maintaining aspect ratio.

    Adds black padding to fill the remaining space.
    Returns JPEG bytes.
    """
    if isinstance(image, str) and not is_url(image):
        img = Image.open(image)
    elif isinstance(image, bytes):
        img = Image.open(io.BytesIO(image))
    else:
        raise ValueError("Unsupported image input")

    if img.mode != "RGB":
        img = img.convert("RGB")

    img_ratio = img.width / img.height
    target_ratio = target_size[0] / target_size[1]

    if img_ratio > target_ratio:
        new_width = target_size[0]
        new_height = int(target_size[0] / img_ratio)
    else:
        new_height = target_size[1]
        new_width = int(target_size[1] * img_ratio)

    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    img_padded = ImageOps.pad(img, target_size, color=(0, 0, 0), centering=(0.5, 0.5))

    buf = io.BytesIO()
    img_padded.save(buf, format="JPEG")
    return buf.getvalue()


def encode_image(
    image_path: str,
    resize: bool = False,
    target_size: tuple[int, int] = (512, 512),
) -> str:
    """
    Encode an image to base64.

    If resize=True, the image is resized with padding before encoding.
    Supports both local file paths and URLs.
    """
    if resize:
        if is_url(image_path):
            img_bytes = resize_with_padding(requests.get(image_path).content, target_size)
        else:
            img_bytes = resize_with_padding(image_path, target_size)
        return base64.b64encode(img_bytes).decode("utf-8")

    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
