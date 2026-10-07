# Hermes Local LLM Polished STT

A Hermes Agent speech-to-text provider (`local_llm_polished`) for customizable transcript enhancement. ASR runs locally with faster-whisper or sherpa-onnx Parakeet V3, with long-audio chunking. An optional bounded LLM pass then enhances the transcript, following instructions from a custom Hermes skill (`post_processing.skill`), an inline prompt, or the built-in default.

**The plugin and its full documentation live in [`local-llm-polished/`](local-llm-polished/README.md).** Start there for install, configuration, and the [privacy and data flow](local-llm-polished/README.md#privacy-and-data-flow) notes. Audio stays local; with post-processing on (the default) the transcript goes to the LLM you configure, which may be remote.

```bash
hermes plugins install dmmeteo/hermes-local-llm-polished#local-llm-polished
```

## Repository layout

| Path | What |
|---|---|
| `local-llm-polished/` | the plugin Hermes installs (`plugin.yaml`, code, README) |
| `tests/` | pytest suite; stubs the Hermes APIs, no Hermes checkout needed |
| `scripts/plugin_runtime.py` | inspect or pre-seed the Parakeet dependency runtime |
| `docs/` | dependency model and a Hermes core proposal |

```bash
python -m pytest -q
```

## License

MIT
