from src.core.types import EvidenceItem


def test_evidence_item_wire_schema_has_observable_fields_only():
    item = EvidenceItem(
        timestamp_start=1.0,
        timestamp_end=2.0,
        timecode_start="00:00:01.000",
        timecode_end="00:00:02.000",
        evidence_basis="frame 1 + transcript span",
        selection_reason="matches the requested event",
        provenance="test",
        retrieval_method="window_inspection",
    )
    wire = item.to_dict()
    assert "reasoning" not in wire
    for key in ("evidence_basis", "selection_reason", "provenance", "retrieval_method"):
        assert key in wire
