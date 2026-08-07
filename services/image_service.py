"""Regional image-generation requests with bounded in-memory Base64 decoding."""

from __future__ import annotations

import base64
import binascii
import io
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from clients.cuhk_apim_client import CUHKAPIMClient
from config.model_catalog import Operation, Region, require_model_operation
from utils.errors import ResponseFormatError
from utils.usage import HeaderSummary, normalize_usage

MAX_DECODED_IMAGE_BYTES = 20 * 1024 * 1024
SUPPORTED_IMAGE_FORMATS = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}


@dataclass(frozen=True, slots=True)
class GeneratedImage:
    content: bytes = field(repr=False)
    media_type: str
    extension: str


@dataclass(frozen=True, slots=True)
class ImageResult:
    status_code: int
    requested_model: str
    region: Region
    images: tuple[GeneratedImage, ...]
    size: str
    quality: str
    output_format: str
    usage: dict[str, int]
    headers: HeaderSummary
    latency_ms: int


class ImageService:
    def __init__(self, client: CUHKAPIMClient) -> None:
        self.client = client

    def generate(
        self,
        *,
        region: Region,
        model_id: str,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "low",
        output_format: str = "png",
        n: int = 1,
        starter_mode: bool = True,
    ) -> ImageResult:
        model = require_model_operation(region, model_id, Operation.IMAGE_GENERATION)
        if not prompt.strip():
            raise ValueError("Enter an image prompt before submitting.")
        if starter_mode and n != 1:
            raise ValueError("Starter mode constrains image count to n = 1.")
        if not 1 <= n <= 4:
            raise ValueError("Image count must be between 1 and 4.")
        if size not in {"1024x1024", "1536x1024", "1024x1536"}:
            raise ValueError("Choose a supported image size.")
        if quality not in {"low", "medium", "high"}:
            raise ValueError("Choose low, medium, or high image quality.")
        if output_format not in {"png", "jpeg", "webp"}:
            raise ValueError("Choose png, jpeg, or webp output format.")

        response = self.client.post(
            Operation.IMAGE_GENERATION.value,
            {
                "model": model.model_id,
                "prompt": prompt.strip(),
                "size": size,
                "quality": quality,
                "output_format": output_format,
                "n": n,
            },
        )
        data = response.body.get("data")
        if not isinstance(data, list) or not data:
            raise ResponseFormatError(
                response.status_code,
                "missing_image_data",
                "The image response did not contain image data. Record the sanitized request ID for support.",
                response.headers,
            )
        images: list[GeneratedImage] = []
        for item in data:
            if not isinstance(item, Mapping) or not isinstance(item.get("b64_json"), str):
                raise ResponseFormatError(
                    response.status_code,
                    "missing_b64_json",
                    "The image response was missing b64_json. Record the sanitized request ID for support.",
                    response.headers,
                )
            images.append(
                _decode_image(item["b64_json"], response, output_format, size)
            )
        if len(images) != n:
            raise ResponseFormatError(
                response.status_code,
                "image_count_mismatch",
                "The number of returned images did not match the requested count.",
                response.headers,
            )
        return ImageResult(
            status_code=response.status_code,
            requested_model=model.model_id,
            region=region,
            images=tuple(images),
            size=size,
            quality=quality,
            output_format=output_format,
            usage=normalize_usage(response.body),
            headers=response.headers,
            latency_ms=response.latency_ms,
        )


def _decode_image(
    encoded: str, response: Any, expected_format: str, expected_size: str
) -> GeneratedImage:
    if len(encoded) > (MAX_DECODED_IMAGE_BYTES * 4 // 3) + 8:
        raise ResponseFormatError(
            response.status_code,
            "image_too_large",
            "The encoded image exceeded the safe local size limit.",
            response.headers,
        )
    try:
        content = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise ResponseFormatError(
            response.status_code,
            "malformed_base64",
            "The image response contained malformed Base64 data.",
            response.headers,
        ) from None
    if not content or len(content) > MAX_DECODED_IMAGE_BYTES:
        raise ResponseFormatError(
            response.status_code,
            "invalid_image_size",
            "The decoded image was empty or exceeded the safe local size limit.",
            response.headers,
        )
    try:
        from PIL import Image

        expected_dimensions = tuple(int(value) for value in expected_size.split("x"))
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as image:
                image_format = (image.format or "").lower().replace("jpg", "jpeg")
                dimensions = image.size
                image.verify()
            if image_format not in SUPPORTED_IMAGE_FORMATS:
                raise ValueError("unsupported format")
            if image_format != expected_format or dimensions != expected_dimensions:
                raise ValueError("format or dimensions do not match the request")
            if dimensions[0] * dimensions[1] > 4_000_000:
                raise ValueError("image pixel count exceeds the local limit")
            with Image.open(io.BytesIO(content)) as image:
                image.load()
    except Exception:  # noqa: BLE001 - image decoder boundary returns a sanitized error
        raise ResponseFormatError(
            response.status_code,
            "unsupported_image_type",
            "The decoded content was not a complete image matching the requested PNG, JPEG, or WebP settings.",
            response.headers,
        ) from None
    return GeneratedImage(content, SUPPORTED_IMAGE_FORMATS[image_format], image_format)
