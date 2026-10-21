# NovaTask — Background Task Orchestrator

NovaTask is a lightweight asynchronous job processing utility built for Python and TypeScript services.

## Overview

NovaTask allows distributed workers to poll, lease, and complete batch computational tasks.

- [Getting Started Guide](./docs/getting_started.md)
- [Architecture Details](./docs/architecture.md)
- [API Reference](https://api.novatask.dev/v1/docs)

## Features

- Dynamic worker leasing
- Heartbeat-driven liveness probes
- Configurable failure retry policies
- Real-time event dispatching

## Setup & Configuration

Copy the example environment configuration to configure your deployment:

```bash
cp .env.example .env
```

Refer to the [Deployment Runbook](./docs/deployment.md) for production recommendations.

## Running Tests

Run the test suite using pytest:

```bash
pytest tests/
```
