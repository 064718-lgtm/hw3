## Context
The system will accept telemetry over MQTT and HTTP. Storage should be abstracted behind an adapter interface so tests can use in-memory stores while local runs can use SQLite/Timescale.

### Goals
- Minimal, testable ingestion pipeline
- Clear boundaries between transport, ingestion, and storage

### Decisions
- Use JSON telemetry messages with the following minimal schema: `{ deviceId: string, timestamp: ISO8601, metrics: { temperature?: number, humidity?: number, battery?: number, signalStrength?: number }, metadata?: Record<string,string> }`.
- Validate required fields; reject messages missing deviceId or metrics.
- Storage adapter API: `write(deviceId: string, timestamp: string, metrics: Record<string,number>, metadata?: Record<string,string>)` and `queryRecent(deviceId?: string, limit?: number)`.

### Risks
- Message volume could grow; the in-memory adapter is not suitable for production. Mitigation: document in README and provide SQLite adapter.
