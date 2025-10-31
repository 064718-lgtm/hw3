# Project Context

## Purpose
This repository contains an AIoT (Applied IoT) homework project focused on device telemetry ingestion, processing, and short-term storage for analytics and demonstration. The primary goals are:

- Collect device telemetry via MQTT and optionally HTTP from simulated or real devices.
- Normalize and validate incoming measurements.
- Store time-series data for short-term queries and visualization.
- Provide a small REST API for querying recent telemetry and device metadata.
- Keep the codebase small and easy to evaluate for homework grading.

## Tech Stack
- Node.js 18+ with TypeScript for server and CLI code
- MQTT (e.g., Mosquitto) for device telemetry transport
- HTTP/REST for control and query APIs
- Lightweight time-series storage: SQLite/Timescale (or in-memory for tests)
- Testing: Jest for unit tests, supertest for HTTP endpoint tests
- Linting/Formatting: ESLint and Prettier

## Project Conventions

### Code Style
- Use TypeScript with strict mode enabled (tsconfig "strict": true).
- Follow ESLint recommended rules plus project-specific rules in `.eslintrc.js`.
- Use Prettier for formatting; avoid bike-shedding — run `npm run format` before commits.
- Naming: camelCase for variables/functions, PascalCase for types/classes, kebab-case for filenames and CLI commands.

### Architecture Patterns
- Small service oriented design: a single process handling ingestion, validation, and short-term storage.
- Clean boundaries: `transport` (MQTT/HTTP), `ingestion` (parsing/validation), `storage` (time-series adapter), and `api` (REST endpoints).
- Dependency inversion for storage adapters so tests can use in-memory stores.

### Testing Strategy
- Unit tests for parsing, validation, and storage adapter logic (Jest).
- Integration tests spin up an in-memory storage and perform end-to-end MQTT->API flows where feasible.
- Keep tests fast: prefer mocking external services unless exercising integration behavior.

### Git Workflow
- Branching: `main` is the stable submission branch. Create feature branches named `feat/<short-desc>` or change branches named `change/<change-id>` for OpenSpec proposals.
- Commits: small, atomic commits. Use Conventional Commits (type(scope): summary) for clarity.
- PRs: include links to `openspec/changes/<change-id>/proposal.md` when implementing a proposal.

## Domain Context
- Devices publish telemetry messages containing: deviceId, timestamp, metrics (temperature, humidity, battery, signalStrength), and optional metadata.
- Telemetry messages may arrive out of order or with duplicate timestamps; ingestion should tolerate minor disorder and deduplicate where obvious.
- For homework purposes, sampling rates are low (seconds to minutes) and throughput is modest.

## Important Constraints
- Keep dependencies minimal and permissive (MIT/Apache-2.0) where possible.
- Do not include heavy cloud-specific SDKs; the project should run locally for grading.
- Security: do not store sensitive keys in the repository; use environment variables for secrets in local runs.

## External Dependencies
- MQTT broker (local test broker such as Mosquitto) for ingestion tests and demos.
- Optional: TimescaleDB or SQLite for time-series storage in real runs. Use an in-memory adapter for tests.
- npm registry for Node dependencies.
