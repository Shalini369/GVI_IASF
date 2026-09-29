"""
GVI (Guiding Visually Impaired) Modules Package.
"""

from modules.object_detection import (
    HazardDetector,
    detect_objects,
    ZONE_OVERHEAD,
    ZONE_GROUND,
    PROXIMITY_CLOSE,
    PROXIMITY_MEDIUM,
    PROXIMITY_FAR,
    CAT_TRANSIT,
    CAT_SIGNAGE,
    CAT_SCENE,
)

__all__ = [
    "HazardDetector",
    "detect_objects",
    "ZONE_OVERHEAD",
    "ZONE_GROUND",
    "PROXIMITY_CLOSE",
    "PROXIMITY_MEDIUM",
    "PROXIMITY_FAR",
    "CAT_TRANSIT",
    "CAT_SIGNAGE",
    "CAT_SCENE",
]
