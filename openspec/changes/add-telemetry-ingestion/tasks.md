## 1. Implementation
- [ ] 1.1 Create storage adapter interface and in-memory adapter
- [ ] 1.2 Implement telemetry parser and validator
- [ ] 1.3 Add MQTT transport module with subscription and handler
- [ ] 1.4 Add HTTP ingestion endpoint for POST /telemetry
- [ ] 1.5 Add REST query endpoints: GET /telemetry/recent?deviceId= and GET /devices
- [ ] 1.6 Write unit tests for parser, storage adapter, and API
- [ ] 1.7 Add integration tests for end-to-end ingestion (in-memory storage)
- [ ] 1.8 Update README with run and test instructions

## 2. Release
- [ ] 2.1 Validate new spec with `openspec validate add-telemetry-ingestion --strict`
- [ ] 2.2 Create PR linking this change
