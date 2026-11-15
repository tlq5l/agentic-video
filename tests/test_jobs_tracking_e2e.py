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
