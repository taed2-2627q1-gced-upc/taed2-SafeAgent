from contextlib import contextmanager
from dataclasses import asdict
from importlib.metadata import version

from codecarbon import OfflineEmissionsTracker

from taed2_safeagent.data.common import write_json


@contextmanager
def measure_training(settings, report):
    result = {
        "version": version("codecarbon"),
        "scope": "fitting and epoch validation",
        "country_iso_code": settings["country_iso_code"],
        "country_assumed": settings["country_assumed"],
        "cpu_and_ram": "estimates unless sensor support is separately verified",
        "gpu": "CodeCarbon NVML support when available",
        "status": "unavailable",
    }
    tracker = None
    try:
        tracker = OfflineEmissionsTracker(
            country_iso_code=settings["country_iso_code"],
            measure_power_secs=settings["measure_power_secs"],
            gpu_ids=settings["gpu_ids"],
            tracking_mode="process",
            save_to_file=False,
            log_level="warning",
        )
        tracker.start()
    except (ValueError, OSError, RuntimeError) as error:
        result["reason"] = type(error).__name__
        tracker = None
    succeeded = False
    try:
        yield
        succeeded = True
    finally:
        result["training_succeeded"] = succeeded
        if tracker is not None:
            try:
                tracker.stop()
                data = tracker.final_emissions_data
                if data is not None:
                    result.update(status="recorded", measurement=asdict(data))
            except (ValueError, OSError, RuntimeError) as error:
                result["reason"] = type(error).__name__
        write_json(report, result)
