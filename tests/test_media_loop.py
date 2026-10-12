"""Synthetic end-to-end tests for the check/judge/fix/re-check loop."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import urllib.error
import wave
from pathlib import Path

import numpy as np
import pytest

from src import jev
from src.loop import _fix_brief, _judge_agreement, _judge_candidates, _markdown_report, run_loop


RATE = 48000


def _write_click_audio(path: Path, duration: float, click_times) -> Path:
    samples = np.zeros(int(duration * RATE), dtype=np.float32)
    count = int(0.025 * RATE)
    t = np.arange(count, dtype=np.float32) / RATE
    pulse = 0.75 * np.sin(2 * np.pi * 1750 * t) * np.exp(-t * 80.0)
    for when in click_times:
        first = int(round(when * RATE))
        samples[first:first + count] += pulse
    pcm = np.clip(samples, -0.95, 0.95)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes((pcm * 32767).astype("<i2").tobytes())
    return path


def _synthetic_video(tmp_path: Path) -> Path:
    audio = _write_click_audio(tmp_path / "planted.wav", 3.5, [0.7, 2.25])
    video = tmp_path / "planted.mp4"
    subprocess.run([
        "ffmpeg", "-nostdin", "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc=size=160x90:rate=24:duration=3.5",
        "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "mpeg4", "-q:v", "5", "-c:a", "aac", "-shortest", str(video),
    ], check=True, capture_output=True, text=True)
    return video


def test_check_finds_planted_missing_and_late_cues_with_evidence(tmp_path):
    video = _synthetic_video(tmp_path)
    cues = tmp_path / "cues.json"
    cues.write_text(json.dumps({"cues": [
        {"time_sec": 1.35, "label": "missing hit"},
        {"time_sec": 1.95, "label": "late hit"},
    ]}), encoding="utf-8")
    env = os.environ.copy()
    command = subprocess.run([
        sys.executable, "-m", "src.loop", "check", str(video), "--cues", str(cues),
        "--out", str(tmp_path / "check"), "--judge", "none",
    ], capture_output=True, text=True, env=env, check=False)
    assert command.returncode == 1
    report = json.loads((tmp_path / "check" / "report.json").read_text(encoding="utf-8"))
    cue_problems = [item for item in report["candidates"] if item["kind"].startswith("cue_")]
    statuses = {item["kind"] for item in cue_problems}
    assert "cue_missing" in statuses
    assert "cue_late" in statuses
    assert all(item["confirmed"] and item["rule_verdict"] == "fault" for item in cue_problems)
    assert all(Path(tmp_path / "check" / item["evidence"]["spectrogram"]).is_file() for item in cue_problems)
    assert all(Path(tmp_path / "check" / item["evidence"]["contact_sheet"]).is_file() for item in cue_problems)
    assert (tmp_path / "check" / "report.md").is_file()


class _Response:
    status = 200

    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.payload


def test_jev_client_parses_response_and_falls_back(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "synthetic-typesafe-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-openrouter-secret")
    seen = []

    def fake_urlopen(request, timeout):
        seen.append((request.full_url, timeout, json.loads(request.data)))
        if "typesafe.ai" in request.full_url:
            raise urllib.error.HTTPError(
                request.full_url, 503, "temporary", {}, io.BytesIO(b"try fallback"),
            )
        return _Response({"answers": {"q1": {"choice": "fault", "confidence": 0.91}}})

    result = jev.decide({"candidate": 1}, {"q1": {"type": "choice"}}, urlopen=fake_urlopen)
    assert result["answers"]["q1"] == {"choice": "fault", "confidence": 0.91}
    assert result["provider"] == "openrouter"
    assert len(seen) == 2
    assert seen[0][2]["model"] == "jev-latest"
    assert seen[1][2]["model"] == "~typesafe/jev-latest"


def test_jev_error_and_report_scrub_configured_keys(monkeypatch):
    secret = "synthetic-typesafe-private-token"
    monkeypatch.setenv("TYPESAFE_API_KEY", secret)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def failing_urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "unauthorized", {}, io.BytesIO(secret.encode()))

    try:
        jev.decide({}, {}, urlopen=failing_urlopen)
    except Exception as exc:
        error = str(exc)
    else:
        pytest.fail("Jev request unexpectedly succeeded")
    if secret in error:
        pytest.fail("credential leaked in client error")

    candidates = [{
        "id": "example", "kind": "cue_missing", "start_sec": 1.0, "end_sec": 1.0,
        "numbers": {"offset_ms": None}, "classification": "sync missing", "shape": None,
        "nearby_cues": [], "rule_verdict": "fault", "_marks": [],
    }]

    def rejected(state, questions):
        raise RuntimeError("provider echoed " + secret)

    notice = _judge_candidates(candidates, "author notes", judge="jev", client=rejected)
    report_text = json.dumps({"candidates": candidates, "judge_notice": notice})
    if secret in report_text:
        pytest.fail("credential leaked in report")
    assert candidates[0]["confirmed"] is True


def test_ambiguous_items_are_separate_from_confirmed_problems_in_fixer_brief(tmp_path):
    candidates = [
        {
            "id": "late-whoosh", "kind": "cue_late", "start_sec": 2.0, "end_sec": 2.0,
            "numbers": {"offset_ms": 250.0, "tolerance_ms": 80.0},
            "nearby_cues": [{"time_sec": 2.0, "label": "cut whoosh 03"}],
            "rule_verdict": "fault", "classification": "cue match late", "evidence": {},
        },
        {
            "id": "unclear-layer", "kind": "cue_ambiguous", "start_sec": 4.0, "end_sec": 4.0,
            "numbers": {"offset_ms": None, "tolerance_ms": 80.0},
            "nearby_cues": [{"time_sec": 4.0, "label": "text pop | line"}],
            "rule_verdict": "ambiguous", "classification": "cue match ambiguous", "evidence": {},
        },
        {
            "id": "planned-ramp", "kind": "loudness_ramp", "start_sec": 6.0, "end_sec": 7.0,
            "numbers": {"delta_db": 3.0}, "nearby_cues": [],
            "rule_verdict": "informational", "classification": "informational", "evidence": {},
        },
    ]
    seen = {}

    def rubber_stamp(state, questions):
        seen.update(questions)
        return {"answers": {qid: {"choice": "real fault", "confidence": 0.70}
                            for qid in questions}}

    _judge_candidates(candidates, "The music ramp is deliberate.", judge="jev", client=rubber_stamp)
    assert [item["confirmed"] for item in candidates] == [True, False, False]
    assert "250.0 ms after" in seen["candidate_1"]["instructions"]
    assert "tolerance is 80.0 ms" in seen["candidate_1"]["instructions"]
    assert "The music ramp is deliberate." in seen["candidate_1"]["instructions"]
    assert set(seen["candidate_1"]["criteria"]) == {
        "real fault", "intentional per the author's notes", "measurement unreliable",
    }
    agreement = _judge_agreement(candidates)
    assert agreement == {"agreed": 1, "total": 3, "rate": pytest.approx(1 / 3, abs=0.001), "rubber_stamp": True}
    markdown = _markdown_report({"video": "synthetic.mp4", "candidates": candidates,
                                 "judge_agreement": agreement})
    assert "## Needs review" in markdown and "Jev agreement with rules" in markdown
    # Ambiguous candidates are not promoted to confirmed problems, but they
    # can be sent in the separate review section.
    candidates[1]["confirmed"] = True
    review = dict(candidates[1], confirmed=False, ambiguous_reason="could be an overlapping sound",
                  _out_dir=str(tmp_path), evidence={
                      "contact_sheet": "evidence/title.jpg", "spectrogram": "evidence/title.png",
                  })
    brief = _fix_brief(candidates, tmp_path, review_items=[review])
    assert "late-whoosh" in brief
    assert "unclear-layer" in brief and "planned-ramp" not in brief
    assert "could be an overlapping sound" in brief and '"offset_ms": null' in brief
    assert str(tmp_path / "evidence" / "title.jpg") in brief
    assert str(tmp_path / "evidence" / "title.png") in brief
    assert "open the frames/evidence" in brief and "measure with ffmpeg" in brief
    assert "REVIEW <id>: fault|not a fault — <short reason>" in brief


def _fake_checker_from_issue_file(issue_path: Path):
    def check(video, **kwargs):
        issues = json.loads(issue_path.read_text(encoding="utf-8"))
        return {"candidates": [
            {
                "id": name, "kind": "synthetic_fault", "start_sec": index + 0.5,
                "end_sec": index + 0.5, "confirmed": True,
                "numbers": {"value": index}, "evidence": {"spectrogram": "evidence/{}.png".format(name)},
            }
            for index, name in enumerate(issues)
        ]}
    return check


def _review_item(item_id="near-edge-title", *, numbers=None):
    return {
        "id": item_id, "kind": "text_cropped", "start_sec": 2.0, "end_sec": 2.8,
        "confirmed": False, "rule_verdict": "ambiguous", "classification": "text may be cropped",
        "ambiguous_reason": "the line ends close to the right edge",
        "numbers": numbers or {"margin_px": 6, "side": "right", "near_edge_positions": 17},
        "evidence": {"contact_sheet": "evidence/title.jpg", "spectrogram": "evidence/title.png"},
    }


def _checker_with_review(issue_path: Path, review=None):
    fault_check = _fake_checker_from_issue_file(issue_path)

    def check(video, **kwargs):
        report = fault_check(video, **kwargs)
        report["candidates"].append(dict(review or _review_item()))
        return report

    return check


def _write_fixer(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "fake_fixer.py"
    script.write_text("import sys\n" + body, encoding="utf-8")
    return script


def _write_config(tmp_path: Path, *, fixer_cmd, build_cmd="true", max_rounds=6, ladder=None):
    config = {
        "project_dir": str(tmp_path),
        "build_cmd": build_cmd,
        "video": "render.mp4",
        "judge": "none",
        "max_rounds": max_rounds,
        "fixer_cmd": fixer_cmd,
        "out": str(tmp_path / "loop-output"),
    }
    if ladder is not None:
        config["ladder"] = ladder
    path = tmp_path / "loop.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_check_cli_and_run_config_default_judge_to_none(tmp_path, monkeypatch):
    import src.loop as loop_module

    seen = []

    def clean_check(video, **kwargs):
        seen.append(kwargs.get("judge"))
        return {"candidates": []}

    monkeypatch.setattr(loop_module, "check_video", clean_check)
    assert loop_module.main(["check", "unused.mp4", "--out", str(tmp_path / "check")]) == 0
    assert seen == ["none"]

    config = _write_config(tmp_path, fixer_cmd=[sys.executable, "-c", "pass"])
    data = json.loads(config.read_text())
    data.pop("judge")
    config.write_text(json.dumps(data), encoding="utf-8")
    result = run_loop(str(config), check_function=clean_check)
    assert result["status"] == "success"
    assert seen == ["none", "none"]


def test_review_only_run_gets_one_fixer_round_and_records_verdict(tmp_path):
    brief_file = tmp_path / "brief.txt"
    fixer = _write_fixer(tmp_path, (
        "from pathlib import Path\n"
        "Path(sys.argv[1]).write_text(sys.stdin.read())\n"
        "print(' review near-edge-title : Not A Fault - complete letters end before the gap ')\n"
    ))
    config = _write_config(tmp_path, fixer_cmd=[sys.executable, str(fixer), str(brief_file)])

    def check(video, **kwargs):
        return {"candidates": [_review_item()]}

    result = run_loop(str(config), check_function=check)
    assert result["status"] == "success"
    assert result["rounds"] == 1
    brief = brief_file.read_text(encoding="utf-8")
    assert "Review items:" in brief and "near-edge-title" in brief
    assert str(tmp_path / "loop-output" / "round-01" / "evidence" / "title.jpg") in brief
    [entry] = [json.loads(line) for line in (tmp_path / "loop-output" / "loop_log.jsonl").read_text().splitlines()]
    assert entry["review_verdicts"] == [{
        "id": "near-edge-title", "verdict": "not a fault",
        "reason": "complete letters end before the gap",
    }]
    assert result["review_verdicts"] == [dict(entry["review_verdicts"][0], round=1)]
    summary = (tmp_path / "loop-output" / "summary.md").read_text(encoding="utf-8")
    assert "## Fixer verdicts on review items" in summary
    assert "**not a fault** — complete letters end before the gap" in summary


def test_not_a_fault_review_is_not_resent_with_unchanged_numbers(tmp_path):
    issues = tmp_path / "issues.json"
    issues.write_text(json.dumps(["persistent"]), encoding="utf-8")
    briefs = tmp_path / "briefs"
    briefs.mkdir()
    calls = tmp_path / "calls"
    fixer = _write_fixer(tmp_path, (
        "from pathlib import Path\n"
        "brief = sys.stdin.read()\n"
        "count = int(Path(sys.argv[2]).read_text()) if Path(sys.argv[2]).exists() else 0\n"
        "Path(sys.argv[1], f'{count}.txt').write_text(brief)\n"
        "Path(sys.argv[2]).write_text(str(count + 1))\n"
        "if count == 0:\n"
        "    print('REVIEW near-edge-title: not a fault — complete text')\n"
        "else:\n"
        "    Path(sys.argv[3]).write_text('[]')\n"
    ))
    config = _write_config(tmp_path, fixer_cmd=[sys.executable, str(fixer), str(briefs), str(calls), str(issues)])

    result = run_loop(str(config), check_function=_checker_with_review(issues))
    assert result["status"] == "success" and result["rounds"] == 2
    first = (briefs / "0.txt").read_text(encoding="utf-8")
    second = (briefs / "1.txt").read_text(encoding="utf-8")
    assert "near-edge-title" in first
    assert "near-edge-title" not in second
    entries = [json.loads(line) for line in (tmp_path / "loop-output" / "loop_log.jsonl").read_text().splitlines()]
    assert entries[0]["review_verdicts"][0]["verdict"] == "not a fault"
    assert entries[1]["review_items_before"] == []


def test_run_loop_fixes_one_problem_per_fixer_call(tmp_path):
    issues = tmp_path / "issues.json"
    issues.write_text(json.dumps(["first", "second"]), encoding="utf-8")
    fixer = _write_fixer(tmp_path, (
        "from pathlib import Path\n"
        "path = Path(sys.argv[1])\n"
        "Path(sys.argv[2]).write_text(sys.stdin.read())\n"
        "items = __import__('json').loads(path.read_text())\n"
        "if items: items.pop(0)\n"
        "path.write_text(__import__('json').dumps(items))\n"
    ))
    brief_file = tmp_path / "brief.txt"
    config = _write_config(tmp_path, fixer_cmd=[sys.executable, str(fixer), str(issues), str(brief_file)])
    result = run_loop(str(config), check_function=_fake_checker_from_issue_file(issues))
    assert result["status"] == "success"
    assert result["rounds"] == 2
    assert json.loads(issues.read_text()) == []
    entries = [json.loads(line) for line in (tmp_path / "loop-output" / "loop_log.jsonl").read_text().splitlines()]
    assert len(entries) == 2
    assert len(entries[0]["problems_before"]) == 2
    assert len(entries[0]["problems_after"]) == 1
    assert entries[1]["problems_after"] == []
    brief = brief_file.read_text()
    assert "Do not edit cues or intent files." in brief
    assert "numbers=" in brief and "evidence=" in brief


def test_run_loop_retries_then_climbs_ladder_to_human_review(tmp_path):
    issues = tmp_path / "issues.json"
    issues.write_text(json.dumps(["stuck"]), encoding="utf-8")
    fixer = _write_fixer(tmp_path, "sys.stdin.read()\n")
    ladder = [{"model": "m{}".format(index), "effort": "e"} for index in range(4)]
    config = _write_config(
        tmp_path, fixer_cmd=[sys.executable, str(fixer)], max_rounds=20, ladder=ladder,
    )
    result = run_loop(str(config), check_function=_fake_checker_from_issue_file(issues))
    assert result["status"] == "needs human review"
    assert result["rounds"] == 8
    assert result["rungs_used"] == ["m0 e", "m0 e", "m1 e", "m1 e", "m2 e", "m2 e", "m3 e", "m3 e"]
    assert "needs human review" in (tmp_path / "loop-output" / "summary.md").read_text().lower()


def test_run_loop_feeds_build_failure_to_fixer(tmp_path):
    sentinel = tmp_path / "build-ok"
    build_script = tmp_path / "build.py"
    build_script.write_text(
        "from pathlib import Path\nimport sys\n"
        "sentinel = Path(sys.argv[1])\n"
        "raise SystemExit(0 if sentinel.exists() else 1)\n",
        encoding="utf-8",
    )
    received = tmp_path / "fixer-input.txt"
    fixer = _write_fixer(tmp_path, (
        "from pathlib import Path\n"
        "Path(sys.argv[1]).write_text(sys.stdin.read())\n"
        "Path(sys.argv[2]).touch()\n"
    ))
    config = _write_config(
        tmp_path,
        build_cmd=[sys.executable, str(build_script), str(sentinel)],
        fixer_cmd=[sys.executable, str(fixer), str(received), str(sentinel)],
    )

    def clean_check(video, **kwargs):
        return {"candidates": []}

    result = run_loop(str(config), check_function=clean_check)
    assert result["status"] == "success"
    assert "Build failed" in received.read_text()
    assert result["rounds"] == 1


@pytest.mark.skipif(os.environ.get("AGENTIC_VIDEO_LIVE_JEV") != "1", reason="opt-in live Jev smoke test")
def test_live_jev_smoke():
    result = jev.decide(
        {"purpose": "live connectivity smoke test"},
        {"connectivity": {
            "type": "choice",
            "instructions": "Return the option that means the request reached Jev.",
            "criteria": {"connected": "The request reached Jev and it can answer."},
        }},
        timeout=15.0,
    )
    if result.get("answers", {}).get("connectivity", {}).get("choice") != "connected":
        pytest.fail("Jev smoke response did not contain the expected choice")


def test_intent_ranges_parse_common_phrasings():
    from src.loop import _parse_intent_sections

    notes = (
        "The mix is muffled from 12.5 s to 15.0 s on purpose. "
        "Underwater bit 20–22.5 s. Between 30 and 31.5 seconds the drums are filtered.\n"
        "Low-pass 0:40 - 0:42.25, deliberately. Riser 50500 ms to 51000 ms. "
        "Bars 9 to 12 build. A 3-6 dB dip is fine. Kicks 4-8 are loud."
    )
    ranges = [(item["start_sec"], item["end_sec"]) for item in _parse_intent_sections(notes)]
    assert ranges == [(12.5, 15.0), (20.0, 22.5), (30.0, 31.5), (40.0, 42.25), (50.5, 51.0)]
    first = _parse_intent_sections(notes)[0]
    assert first["sentence"] == "The mix is muffled from 12.5 s to 15.0 s on purpose."


def _sync_cue(time_sec, label, status, **extra):
    cue = {"time_sec": time_sec, "label": label, "status": status, "offset_ms": None,
           "cue_group": label.split()[0], "audible_in_any_band": True}
    cue.update(extra)
    return cue


def test_intent_section_cues_are_informational_unless_silent_in_every_band():
    from src.loop import _build_candidates, _relevant_intent

    intent = "Punchy intro.\nFrom 12.5 s to 15.0 s the mix is deliberately muffled.\nClean ending."
    sync = {"tolerance_ms": 80.0, "cues": [
        _sync_cue(13.0, "pop 1", "missing"),
        _sync_cue(14.0, "pop 2", "missing", audible_in_any_band=False),
        _sync_cue(16.0, "pop 3", "missing"),
    ]}
    candidates = {item["cue_label"]: item for item in _build_candidates(sync, {}, {}, [], intent=intent)}
    assert candidates["pop 1"]["rule_verdict"] == "informational"
    assert candidates["pop 2"]["rule_verdict"] == "fault"
    assert candidates["pop 3"]["rule_verdict"] == "fault"
    # Jev sees the one matching sentence, not the whole notes file.
    assert _relevant_intent(intent, candidates["pop 1"]) == "From 12.5 s to 15.0 s the mix is deliberately muffled."


def test_duplicate_cues_and_rows_collapse(tmp_path):
    from src.loop import _build_candidates, _load_cues

    path = tmp_path / "cues.json"
    path.write_text(json.dumps({"cues": [
        {"t": 1.0, "label": "Kick  beat 1"}, {"time_sec": 1.0004, "label": "kick beat 1 "},
        {"t": 1.0, "label": "whoosh"}, {"t": 2.0, "label": "kick beat 1"},
    ]}), encoding="utf-8")
    assert [(cue.get("t", cue.get("time_sec")), cue["label"]) for cue in _load_cues(str(path))] == [
        (1.0, "Kick  beat 1"), (1.0, "whoosh"), (2.0, "kick beat 1"),
    ]
    sync = {"tolerance_ms": 80.0, "cues": [
        _sync_cue(3.0, "late hit", "ambiguous"), _sync_cue(3.0, "Late  hit", "late", offset_ms=150.0),
    ], "systematic_lags": [{"group": "late", "lag_ms": 120, "method": "group_template_attack"}]}
    rows = _build_candidates(sync, {}, {}, [], intent="")
    labelled = [(item["cue_label"], item["start_sec"]) for item in rows]
    assert len(labelled) == len(set(labelled)), labelled
    cue_rows = [item for item in rows if item["kind"].startswith("cue_") and item["kind"] != "cue_group_systematic_lag"]
    assert [item["kind"] for item in cue_rows] == ["cue_late"]
    assert len({item["id"] for item in rows}) == len(rows)


def test_intent_ranges_are_half_open_at_their_end():
    from src.loop import _build_candidates, _relevant_intent

    intent = "Muffled from 12.5 s through 15.000 s. Then the drop."
    sync = {"tolerance_ms": 40.0, "cues": [
        _sync_cue(12.5, "hit start", "missing"),
        _sync_cue(14.9, "hit inside", "missing"),
        _sync_cue(14.97, "hit near end", "missing"),
        _sync_cue(15.0, "hit end", "missing"),
    ]}
    candidates = {item["cue_label"]: item for item in _build_candidates(sync, {}, {}, [], intent=intent)}
    assert candidates["hit start"]["rule_verdict"] == "informational"
    assert candidates["hit inside"]["rule_verdict"] == "informational"
    # Within one tolerance of the end, or at it, the next section applies.
    assert candidates["hit near end"]["rule_verdict"] == "fault"
    assert candidates["hit end"]["rule_verdict"] == "fault"
    assert "intent_note" not in candidates["hit end"]
    assert "Muffled" not in _relevant_intent(intent, candidates["hit end"])
