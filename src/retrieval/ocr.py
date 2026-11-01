"""Offline on-screen text recognition through Apple Vision (macOS, via pyobjc)."""
import io
import re
import unicodedata

TEXT_LINE_CAP = 40


def _unavailable(feature, alternative):
    return (f"{feature} needs Apple Vision through pyobjc (macOS only). "
            "Run: python -m pip install -r requirements-visual.txt (adds pyobjc-framework-Vision), "
            f"or {alternative}.")


OCR_UNAVAILABLE = _unavailable("On-screen text matching", "omit `text` and describe the object with `description` instead")
READ_TEXT_UNAVAILABLE = _unavailable("read_text", "leave read_text out (the default) to get the frames without text")


def ensure_available(message=OCR_UNAVAILABLE):
    """Raise an actionable error before any work is scheduled when OCR cannot run."""
    try:
        import Vision  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(message) from exc


def recognize(image):
    """Return [{text, confidence, x, y, w, h}] for a PIL image; boxes are normalized, top-left origin.

    Language correction is off so a near-miss such as RELEASED is never "fixed" into RELEASE.
    """
    ensure_available()
    import objc
    import Vision
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG", compress_level=1)
    with objc.autorelease_pool():
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(False)
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(buffer.getvalue(), {})
        ok, error = handler.performRequests_error_([request], None)
        if not ok:
            raise RuntimeError(f"Apple Vision text recognition failed: {error}")
        lines = []
        for observation in request.results() or []:
            candidates = observation.topCandidates_(1)
            if not candidates:
                continue
            box = observation.boundingBox()
            lines.append({"text": str(candidates[0].string()), "confidence": float(candidates[0].confidence()),
                          "x": float(box.origin.x), "y": float(1.0 - box.origin.y - box.size.height),
                          "w": float(box.size.width), "h": float(box.size.height)})
        return lines


def frame_text(path):
    """text_lines for one saved frame, top to bottom then left to right; at most TEXT_LINE_CAP lines."""
    from PIL import Image
    with Image.open(path) as image:
        lines = sorted(recognize(image), key=lambda l: (l["y"], l["x"]))
    out = {"text_lines": [
        {"text": l["text"], "confidence": round(l["confidence"], 3),
         "box": {"x": round(l["x"], 4), "y": round(l["y"], 4), "width": round(l["w"], 4), "height": round(l["h"], 4)}}
        for l in lines[:TEXT_LINE_CAP]]}
    if len(lines) > TEXT_LINE_CAP:
        out["text_lines_note"] = (f"Lists the first {TEXT_LINE_CAP} of {len(lines)} text lines in this frame "
                                  "(top to bottom); the rest are left out.")
    return out


def _normalize(text, case_sensitive):
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()
    return text if case_sensitive else text.casefold()


def phrase_pattern(phrase, case_sensitive=False):
    """Whole-word pattern: RELEASE matches 'RELEASE' and 'RELEASE NOTES' but not RELEASED or PRE-RELEASE."""
    words = _normalize(phrase, case_sensitive).split(" ")
    return re.compile(r"(?<![\w'’-])" + r"\s+".join(re.escape(w) for w in words) + r"(?![\w'’-])")


def find_phrase(lines, pattern, case_sensitive=False):
    """Best matching line for the phrase, or None. Also tries all lines joined in reading order."""
    best = None
    for line in lines:
        if pattern.search(_normalize(line["text"], case_sensitive)):
            if best is None or line["confidence"] > best["confidence"]:
                best = line
    if best is None and len(lines) > 1:
        ordered = sorted(lines, key=lambda l: (round(l["y"] * 40), l["x"]))
        if pattern.search(_normalize(" ".join(l["text"] for l in ordered), case_sensitive)):
            x0 = min(l["x"] for l in ordered); y0 = min(l["y"] for l in ordered)
            best = {"confidence": min(l["confidence"] for l in ordered), "x": x0, "y": y0,
                    "w": max(l["x"] + l["w"] for l in ordered) - x0,
                    "h": max(l["y"] + l["h"] for l in ordered) - y0}
    return best
