"""End to end: count_video joins brief detection fragments only under the round-8 rules (one fragment under 3 samples, absence at most JOIN_GAP_SEC, centre moved at most TELEPORT_JUMP, a direction passed) and never serves results stored before the join existed. Each test fails with one round-10 surviving mutant (m4 jump check dropped, m5 gap doubled, m8 any-direction joins, m9 join-v1 dropped from the cache key)."""
import hashlib, json, shutil
import pytest
from src.retrieval import occurrence
try:
    from tests.test_occurrences import BLUE, H, W, count, make_clip, shape
except ImportError:
    from test_occurrences import BLUE, H, W, count, make_clip, shape
pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is required")


def _clip(path, extra):
    """A filled blue circle seen at samples 1.0-1.6 s (fps 5) moving right from x .1 to .4, then one more sample `extra` = (t, x)."""
    t1, x1 = extra

    def paint(draw, t):
        if .95 <= t < 1.65:
            shape(draw, "circle", int((.1 + .5 * (t - 1.0)) * W), H // 2, 16, BLUE, outline=False)
        if t1 - .05 <= t < t1 + .05:
            shape(draw, "circle", int(x1 * W), H // 2, 16, BLUE, outline=False)
    return make_clip(path, 3.2, paint)


def test_jump_splits(tmp_path):
    video = _clip(tmp_path / "jump.mp4", (2.2, .95))  # 0.4 s absence, centre moves .55 > TELEPORT_JUMP
    assert count(video, tmp_path, description="blue circle", direction="left_to_right")["count"] == 2


def test_long_gap_splits(tmp_path):
    video = _clip(tmp_path / "gap.mp4", (2.6, .45))  # 0.8 s absence > JOIN_GAP_SEC
    assert count(video, tmp_path, description="blue circle", direction="left_to_right")["count"] == 2


def test_any_direction_does_not_join(tmp_path):
    video = _clip(tmp_path / "any.mp4", (2.2, .45))  # 0.4 s absence, small move: joined only when a direction is passed
    assert count(video, tmp_path / "a", description="blue circle")["count"] == 2
    assert count(video, tmp_path / "b", description="blue circle", direction="left_to_right")["count"] == 1


def test_results_before_join_not_served(tmp_path, monkeypatch):
    video = _clip(tmp_path / "cache.mp4", (2.2, .45))
    payloads, real = [], hashlib.sha256

    def spy(data=b"", *a, **k):
        payloads.append(bytes(data))
        return real(data, *a, **k)
    monkeypatch.setattr(occurrence.hashlib, "sha256", spy)
    first = count(video, tmp_path, description="blue circle", direction="left_to_right")
    monkeypatch.setattr(occurrence.hashlib, "sha256", real)
    payload = next(p for p in payloads if b'"back-v1"' in p)
    stale_payload = payload.replace(b', "join-v1"', b"")
    tag, stale_tag = real(payload).hexdigest()[:20], real(stale_payload).hexdigest()[:20]
    stored = next((tmp_path / "cache").rglob(f"{tag}.occurrences.json"))
    stale = json.loads(stored.read_text())
    stale["count"] = 99
    (stored.parent / f"{stale_tag}.occurrences.json").write_text(json.dumps(stale))
    again = count(video, tmp_path, description="blue circle", direction="left_to_right")
    assert again["count"] == first["count"] == 1
