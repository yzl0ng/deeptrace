# DeepTrace Web

This directory contains the bilingual DeepTrace workbench and its Cloudflare
Worker runtime. For the project architecture, backend, training and evaluation
instructions, read the [repository README](../../README.md).

## Local development

```bash
cp .env.example .env
pnpm install --frozen-lockfile
pnpm dev
```

The static architecture and replay pages work without credentials. Live mode
requires `DEEPSEEK_API_KEY`, `LIVE_DEMO_ACCESS_TOKEN`, and the D1 binding named
`DB` declared in `.openai/hosting.json`.

## Validation

```bash
pnpm test
pnpm lint
```

`pnpm test` performs a production build and runs the rendered-page and Worker
contract tests.

## Configuration boundary

- `.env` is local and ignored by Git.
- `.openai/hosting.json` contains only logical binding names and a placeholder
  project ID in the public repository.
- live run records and SSE traces are stored in D1; secrets are never written to
  the browser bundle.
