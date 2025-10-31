## Why
We need a clear, testable telemetry ingestion capability so the project can accept device data (MQTT/HTTP), validate it, and store it for short-term queries and demonstrations.

## What Changes
- **ADDED** new capability: `telemetry` to handle ingestion, validation, and storage of device telemetry.
- Add server-side modules for MQTT and HTTP ingestion.
- Add a storage adapter interface and an in-memory adapter for tests plus SQLite adapter for local runs.
- Add REST endpoints to query recent telemetry and device metadata.

## Impact
- Affected specs: `telemetry` (new)
- Affected code: `src/transport/*`, `src/ingestion/*`, `src/storage/*`, `src/api/*`
- Breaks: none (non-breaking, additive)

