## ADDED Requirements
### Requirement: Telemetry Ingestion
The system SHALL accept device telemetry over MQTT and HTTP and persist validated telemetry to a storage adapter.

#### Scenario: MQTT telemetry ingestion success
- **WHEN** a device publishes a valid telemetry JSON message to the `telemetry/<deviceId>` topic
- **THEN** the ingestion pipeline SHALL validate the payload, store the measurement via the storage adapter, and acknowledge processing (no error)

#### Scenario: HTTP telemetry ingestion success
- **WHEN** a client POSTs a valid telemetry JSON payload to `POST /telemetry`
- **THEN** the ingestion pipeline SHALL validate the payload, store the measurement via the storage adapter, and return HTTP 201

#### Scenario: Invalid telemetry rejected
- **WHEN** a telemetry message lacks `deviceId` or `metrics` or contains non-numeric metric values
- **THEN** the system SHALL reject the message, log the validation error, and not persist the data

### Requirement: Query Recent Telemetry
The system SHALL provide an API endpoint `GET /telemetry/recent?deviceId=&limit=` that returns recent measurements, ordered newest-first.

#### Scenario: Query recent telemetry
- **WHEN** a client requests recent telemetry with `deviceId` and optional `limit`
- **THEN** the system SHALL return up to `limit` records (default 100) for the device, each containing timestamp and metrics
