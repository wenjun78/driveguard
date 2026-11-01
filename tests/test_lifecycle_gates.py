"""Champion promotion hold and the relative retraining floor — the two gates that keep the published results
describing the model that is actually scoring the fleet."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from driveguard import monitor
from driveguard.config import Settings
from driveguard.train import promote_if_better

END_DAY = (date(2025, 3, 31) - date(1970, 1, 1)).days  # champion's held-out test period ends 31 Mar 2025


class FakeClient:
    def __init__(self, champion_version="4", val_pr_auc=0.145, end_day=END_DAY):
        self._champ = SimpleNamespace(version=champion_version, run_id="r4")
        self._run = SimpleNamespace(
            data=SimpleNamespace(metrics={"val_pr_auc": val_pr_auc}, params={"split_end_day": str(end_day)})
        )
        self.aliased: list[tuple[str, str, str]] = []

    def get_model_version_by_alias(self, name, alias):
        if self._champ is None:
            raise RuntimeError("no alias")
        return self._champ

    def get_run(self, run_id):
        return self._run

    def set_registered_model_alias(self, name, alias, version):
        self.aliased.append((name, alias, version))


def test_promotion_is_held_while_replay_is_inside_champion_test_period():
    c = FakeClient()
    msg = promote_if_better(c, "m", "5", "champion", 0.99, now=date(2025, 2, 10))
    assert msg.startswith("held champion v4") and "2025-03-31" in msg
    assert c.aliased == []  # registered as challenger, alias untouched


def test_promotion_resumes_after_the_held_out_period_and_still_needs_to_beat_incumbent():
    c = FakeClient()
    assert promote_if_better(c, "m", "5", "champion", 0.10, now=date(2025, 4, 1)).startswith("kept champion v4")
    assert c.aliased == []
    assert promote_if_better(c, "m", "6", "champion", 0.20, now=date(2025, 4, 1)) == "promoted v6 to @champion"
    assert c.aliased == [("m", "champion", "6")]


def test_first_model_is_promoted_unconditionally():
    c = FakeClient()
    c._champ = None
    assert promote_if_better(c, "m", "1", "champion", 0.01, now=date(2025, 1, 1)) == "promoted v1 to @champion"


def test_recall_floor_is_relative_to_champion_when_known():
    s = Settings(matured_recall_floor_ratio=0.5, matured_recall_floor=0.3)
    floor, basis = monitor.recall_floor(s, {"test_recall_at_budget": 0.453})
    assert abs(floor - 0.2265) < 1e-9 and basis.startswith("0.5 x champion")
    floor, basis = monitor.recall_floor(s, {})
    assert floor == 0.3 and basis == "absolute fallback"


def test_retrain_decision_uses_floor_and_drift_rule():
    s = Settings(psi_features_for_retrain=3)
    # a weak week (0.31 mean) is above half the champion's 0.453 -> no retrain
    retrain, reasons, mean = monitor.retrain_decision([], [0.3, 0.32, 0.31], 0.2265, "0.5 x champion", s)
    assert not retrain and reasons == [] and abs(mean - 0.31) < 1e-9
    # a collapse is caught
    retrain, reasons, _ = monitor.retrain_decision([], [0.1, 0.2, 0.15], 0.2265, "0.5 x champion", s)
    assert retrain and "< floor 0.227" in reasons[0]
    # drift on enough key features triggers on its own
    retrain, reasons, _ = monitor.retrain_decision(["a", "b", "c"], [0.5], 0.2265, "x", s)
    assert retrain and reasons[0].startswith("3 key features drifted")
    assert not monitor.retrain_decision(["a", "b"], [0.5], 0.2265, "x", s)[0]


def test_docs_markers_track_the_champion(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("refresh_docs", Path("scripts/refresh_docs.py"))
    rd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd)

    # the committed docs already match the committed snapshot
    vals = rd.values()
    for f in rd.FILES:
        assert rd.render(f.read_text(), vals) == f.read_text(), f"{f.name} is out of date: run scripts/refresh_docs.py"
    assert rd.MARK.search(rd.FILES[0].read_text()), "README has no markers"

    # a new champion rewrites every marked figure
    snap = tmp_path / "snap"
    snap.mkdir()
    m = json.loads((rd.SNAPSHOT / "model.json").read_text())
    m["champion"]["version"] = 9
    m["report"]["test"]["drive_recall_at_budget"] = 0.8
    m["report"]["test"]["lead_time_median_days"] = 21
    (snap / "model.json").write_text(json.dumps(m))
    v2 = rd.values(snap)
    text = "v<!--dg:champion.version-->4<!--/dg-->, <!--dg:test.drive_recall0-->71<!--/dg--> %, <!--dg:test.lead_median-->18<!--/dg--> d"
    assert (
        rd.render(text, v2)
        == "v<!--dg:champion.version-->9<!--/dg-->, <!--dg:test.drive_recall0-->80<!--/dg--> %, <!--dg:test.lead_median-->21<!--/dg--> d"
    )
    try:
        rd.render("<!--dg:no.such.key-->x<!--/dg-->", v2)
    except KeyError:
        pass
    else:
        raise AssertionError("unknown marker must fail loudly")
