# Vendored DUCO input adapter

Source: https://github.com/unitreerobotics/televuer
Upstream base: b6ed5db
DUCO snapshot: 8579f46dc7a91bddf9dafdad36a21b788abd7c55
Imported from the author's 3090 checkout, 2026-11-17.

This directory was previously a gitlink pointing at author-local commits while its URL still pointed at upstream. It is now tracked directly so that cloning Duco-Teleop includes its custom input wrappers. Python sources are copied unchanged. An empty accidental file named `=1.21` is omitted. The MIT LICENSE is retained.

Custom files: duco_relative_mapper.py, duco_tv_wrapper.py, hand_pose_utils.py, visionpro_tv_wrapper.py, plus package exports in __init__.py.

The native AVP route imports `VisionProStreamer` from the external `avp_stream` package (VisionProTeleop / Tracking Streamer). This dependency and its visionOS application are not bundled here. The exact installed version on the author's device has not been established; no dependency pin is invented.

Reference: https://github.com/Improbable-AI/VisionProTeleop
