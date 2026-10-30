# Contributing

Contributions of all kinds are welcome. Thanks to everyone who helps make this project better!

## Guidelines

- Keep each pull request to one topic.
- Skills (`skills/*/SKILL.md`) are loaded into the agent's context on every run. Keep them short, imperative.
- MCP tools live in `src/opencine/mcp/server.py`. Each one returns a JSON-able dict, and its docstring is what the agent reads, so keep it short and exact.
- Code that runs on a Space goes in `src/opencine/gen/spaces/` as real files, not in Python strings (see its README).
- New settings use the `OPENCINE_` prefix (`src/opencine/core/config.py`). Add secret ones to `.env.example` with an empty value.
- Never commit tokens, `.env` files, footage or project folders.
- If you add a model or Space, list it under Acknowledgements in the README with its licence.

## Issues

Open an issue at [github.com/tranlamm/open-cine/issues](https://github.com/tranlamm/open-cine/issues) with your OS, your agent, what you asked and the error text. Report security problems privately via [GitHub security advisories](https://github.com/tranlamm/open-cine/security/advisories/new).

## License

By contributing, you agree that your work is released under the [Apache License 2.0](LICENSE).
