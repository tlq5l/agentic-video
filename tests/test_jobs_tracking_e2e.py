"""End to end through the job tracker the stdio server uses (src/core/jobs.py). finish() must wait for a job that
another job starts as it ends, and track() must keep a thread that another caller has not started yet."""

import threading
import time

from src.core import jobs


def test_finish_waits_for_a_job_that_a_job_starts_as_it_ends():
    started = {}

    def second_job():
        time.sleep(1.0)

    def first_job():
        time.sleep(0.3)  # still running when finish() takes its snapshot
        nxt = jobs.track(threading.Thread(target=second_job, name="second-job", daemon=True))
        started["second"] = nxt
        nxt.start()

    jobs.track(threading.Thread(target=first_job, name="first-job", daemon=True)).start()
    alive = jobs.finish()
    assert not started["second"].is_alive(), "finish() returned while a job it did not wait for was still running"
    assert alive == [], alive


def test_track_keeps_a_thread_another_caller_has_not_started_yet():
    done = []

    def slow_job():
        time.sleep(1.5)
        done.append(True)

    waiting = jobs.track(threading.Thread(target=slow_job, name="not-started-yet", daemon=True))
    # Another caller's track() runs before this thread starts; it must not drop the thread from the tracked list.
    jobs.track(threading.Thread(target=lambda: None, name="other", daemon=True)).start()
    waiting.start()
    alive = jobs.finish()
    assert not waiting.is_alive(), "finish() returned while the thread that was not started yet was still running"
    assert alive == [] and done == [True], (alive, done)


def test_finish_waits_for_a_tracked_thread_that_starts_after_finish_begins():
    done = []

    def job():
        time.sleep(0.5)
        done.append(True)

    late = jobs.track(threading.Thread(target=job, name="late-start", daemon=True))
    threading.Timer(0.3, late.start).start()  # not started when finish() begins
    alive = jobs.finish()
    assert not late.is_alive() and done == [True], "finish() returned before a tracked thread that started later had ended"
    assert alive == [], alive


def test_a_start_that_fails_does_not_make_finish_wait_for_its_bound(monkeypatch):
    # The shadow-trial caller tracks its thread, then start() raises; the caller swallows that failure. The thread never ran,
    # so finish() must not hold the server for its whole bound waiting for it.
    from src.transcript import shadow_cf_asr

    monkeypatch.setenv(shadow_cf_asr.SWITCH_ENV, "test-host")
    monkeypatch.setattr(shadow_cf_asr.socket, "gethostname", lambda: "test-host")

    def refuse_to_start(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", refuse_to_start)
    shadow_cf_asr.start_in_background(source_path="unused.mp4")  # never raises
    monkeypatch.undo()

    began = time.monotonic()
    alive = jobs.finish(bound_sec=5)
    elapsed = time.monotonic() - began
    assert alive == [] and elapsed < 1.0, (alive, round(elapsed, 2))
