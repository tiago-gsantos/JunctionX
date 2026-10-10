"""Check an official-format annotations/predictions dict for completeness and valid values."""
from dataclasses import dataclass, field

from shelfmon.planogram import Area

LEVEL_VALUES = {"empty": 0, "half": 0.5, "full": 1}
KEYS = {"planogram_compliant", "stock_level", "stock_value"}


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)  # would lose points or break the format
    warnings: list[str] = field(default_factory=list)  # valid, but worth a look
    coverage: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)  # cam -> images, of, areas, of


def check_entry(entry) -> list[str]:
    """Problems with one area's values (empty list when valid)."""
    if not isinstance(entry, dict):
        return [f"entry is {type(entry).__name__}, not an object"]
    problems = [f"missing key '{k}'" for k in sorted(KEYS - entry.keys())]
    problems += [f"unexpected key '{k}'" for k in sorted(entry.keys() - KEYS)]
    if "planogram_compliant" in entry and not isinstance(entry["planogram_compliant"], bool):
        problems.append(f"planogram_compliant must be true/false, got {entry['planogram_compliant']!r}")
    level, value = entry.get("stock_level"), entry.get("stock_value")
    if "stock_level" in entry and level not in LEVEL_VALUES:
        problems.append(f"stock_level must be empty/half/full, got {level!r}")
    if "stock_value" in entry and (isinstance(value, bool) or value not in LEVEL_VALUES.values()):
        problems.append(f"stock_value must be 0/0.5/1, got {value!r}")
    elif level in LEVEL_VALUES and value != LEVEL_VALUES[level]:
        problems.append(f"stock_value {value!r} does not match stock_level '{level}'")
    return problems


def verify(annotations: dict, planogram: dict[str, list[Area]], images: dict[str, list[str]]) -> Report:
    """Every planogram area of every expected image must be present with valid values.

    `images` maps camera id -> the image file names that must be covered.
    """
    report = Report()
    for cam in sorted(annotations.keys() - planogram.keys()):
        report.errors.append(f"{cam}: camera not in planogram")

    for cam, areas in planogram.items():
        expected_images = images.get(cam, [])
        given = annotations.get(cam, {})
        area_ids = {a.area_id for a in areas}
        for image in sorted(given.keys() - set(expected_images)):
            report.errors.append(f"{cam}/{image}: image not in the image folder")

        complete_images = present_areas = 0
        for image in expected_images:
            entries = given.get(image)
            if entries is None:
                report.errors.append(f"{cam}/{image}: image missing ({len(area_ids)} areas, -5 each)")
                continue
            missing = sorted(area_ids - entries.keys(), key=lambda a: int(a.rsplit("_", 1)[1]))
            if missing:
                report.errors.append(f"{cam}/{image}: {len(missing)} area(s) missing: {', '.join(missing)}")
            else:
                complete_images += 1
            for area_id in sorted(entries.keys() - area_ids):
                report.errors.append(f"{cam}/{image}/{area_id}: area not in planogram for camera {cam}")
            for area_id, entry in entries.items():
                if area_id not in area_ids:
                    continue
                present_areas += 1
                for problem in check_entry(entry):
                    report.errors.append(f"{cam}/{image}/{area_id}: {problem}")
                if isinstance(entry, dict) and entry.get("stock_level") == "empty" \
                        and entry.get("planogram_compliant") is False:
                    report.warnings.append(f"{cam}/{image}/{area_id}: empty but marked non-compliant "
                                           "(convention: empty => compliant)")
        report.coverage[cam] = (complete_images, len(expected_images), present_areas,
                                len(expected_images) * len(area_ids))
    return report
