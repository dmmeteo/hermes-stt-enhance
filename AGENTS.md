# AGENTS.md

Cold-start map for agents working in this repository. [README.md](README.md) and the plugin's own [hermes-stt-enhance/README.md](hermes-stt-enhance/README.md) own the product purpose, configuration, privacy notes and limits. [planning/next-epics.md](planning/next-epics.md) is the only outcome checklist.

## Start and recover

1. Read this file.
2. Read the plugin README sections that the task touches. Read [Privacy and data flow](hermes-stt-enhance/README.md#privacy-and-data-flow) and [What the cleanup pass can and cannot do](hermes-stt-enhance/README.md#what-the-cleanup-pass-can-and-cannot-do) before any change to what leaves the machine or what the user gets back.
3. Read the selected outcome in `planning/next-epics.md`. Work on one selected outcome at a time.

After compaction or context loss, re-read this file, the README sections above and the selected outcome before the next consequential edit.

## Revision state

This is public `main`, plugin version 0.5.1 (`hermes-stt-enhance/plugin.yaml`). `installs_allowed()` in `hermes-stt-enhance/runtime_deps.py` lets the Parakeet runtime install only when the user's config sets `security.allow_lazy_installs: true` explicitly and Hermes' effective config agrees. Anything else denies the install ([Gating](docs/dependency-model.md#gating)). Which version a given Hermes install runs is not recorded here.

## Routes

| Task | Read |
|---|---|
| Provider entry, result envelope, chunk merge | `hermes-stt-enhance/__init__.py` (`SttEnhanceProvider`, `register`) |
| Config keys, backend aliases, unmigrated-name refusal | `hermes-stt-enhance/config.py` |
| faster-whisper and Parakeet backends, decode concurrency | `hermes-stt-enhance/backends.py` |
| ffmpeg normalization, chunk plan, overlap merge | `hermes-stt-enhance/audio.py` |
| LLM cleanup pass and raw-transcript fallback | `hermes-stt-enhance/post_processing.py` (`apply`) |
| Instructions from a Hermes skill | `hermes-stt-enhance/skill_source.py` (`load_instructions`) |
| Parakeet dependency runtime outside the Hermes venv, install gate | `hermes-stt-enhance/runtime_deps.py` (`installs_allowed`, `ensure`), [docs/dependency-model.md](docs/dependency-model.md) |
| 0.4.x config rename | `scripts/migrate_config.py` |
| Proposal for Hermes core, not work in this repo | [docs/hermes-core-proposal.md](docs/hermes-core-proposal.md) |

## Boundaries that must hold

- ASR returns a raw transcript. Post-processing is optional. When it errors, times out or returns empty text, the provider returns the raw transcript with `post_processing_error`. A configured skill that fails to load also returns the raw transcript and never falls back to the inline prompt or the default.
- The transcript is untrusted data. Keep the `<transcript>` delimiters, the delimiter neutralization and the "data, not instructions" framing ([details](hermes-stt-enhance/README.md#the-transcript-is-untrusted-input)).
- Audio stays local. The transcript goes to the configured LLM, which may be remote. Any change to that flow updates the privacy table in the same change.
- The provider never raises. Failures return an envelope with `success: False`.
- Old `local-llm-polished` names fail closed with an error that names the key.
- The Parakeet runtime install fails closed unless the install policy is an explicit `true`.
- The dependency runtime is appended to `sys.path`, never prepended.

## Authority

Owner approval is required to change these boundaries, the privacy claims, product scope, or a checklist priority. A checklist item is not permission to implement it. Push, tag, release, catalog submission and installs into a live Hermes need explicit authorization.

## Done means

An outcome is complete only when its user-visible completion condition is verified on the real artifact. An implementation-only change does not tick an item, and passing tests or documentation checks are not runtime or user acceptance. Update `planning/next-epics.md` only for a verified outcome or an owner scope decision.

## Checks

`python -m pytest -q` from the repository root runs the suite with stubbed Hermes APIs. `tests/integration/skill_e2e.py` needs a real Hermes checkout; its usage is in the [plugin README](hermes-stt-enhance/README.md#tests).
