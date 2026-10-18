from f1engine.store import Store
from f1engine.repairs import repair_identities


def test_historical_car_number_repair_uses_event_result_and_is_idempotent(tmp_path):
    store = Store(tmp_path / "repair")
    source = store.raw(
        "jolpica",
        "fixture://old-driver-profile",
        {
            "MRData": {
                "RaceTable": {
                    "Races": [
                        {
                            "season": "2024",
                            "round": "16",
                            "QualifyingResults": [
                                {"Driver": {"driverId": "norris", "permanentNumber": "1"}, "number": "4"},
                                {
                                    "Driver": {"driverId": "max_verstappen", "permanentNumber": "1"},
                                    "number": "1",
                                },
                            ],
                        }
                    ]
                }
            }
        },
    )
    rows = [
        {
            "event_id": "2024-16",
            "driver_id": d,
            "number": "1",
            "source_id": source,
            "available_at": "2024-08-25T00:00:00Z",
            "availability_basis": "reconstructed_entry",
        }
        for d in ("norris", "max_verstappen")
    ]
    old = store.publish({"entries": rows})
    result = repair_identities(store)
    assert result["car_number_corrections"] == 1
    assert {r["driver_id"]: r["number"] for r in store.table("entries")} == {
        "norris": "4",
        "max_verstappen": "1",
    }
    assert all(r["number"] == "1" for r in store.table("entries", old))
    again = repair_identities(store)
    assert again["car_number_corrections"] == 0 and again["data_version"] == result["data_version"]
