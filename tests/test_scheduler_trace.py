import unittest

from nautical_core.scheduler_trace import SchedulerTrace


class SchedulerTraceTests(unittest.TestCase):
    def test_decision_count_is_not_limited_by_event_retention(self) -> None:
        trace = SchedulerTrace(enabled=True, max_events=1)
        trace.record("candidate", candidate="one")
        trace.record("candidate", candidate="two")
        self.assertEqual(trace.decision_count, 2)
        self.assertEqual(len(trace.events), 1)
        self.assertEqual(trace.summary()["decision_count"], 2)
        trace.clear()
        self.assertEqual(trace.decision_count, 0)


if __name__ == "__main__":
    unittest.main()
