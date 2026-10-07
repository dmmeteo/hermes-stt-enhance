# Hermes STT Enhance

A Hermes Agent speech-to-text provider (`stt_enhance`) for customizable transcript enhancement. ASR runs locally with faster-whisper or sherpa-onnx Parakeet V3, with long-audio chunking. An optional bounded LLM pass then enhances the transcript, following instructions from a custom Hermes skill (`post_processing.skill`), an inline prompt, or the built-in default.

**The plugin and its full documentation live in [`hermes-stt-enhance/`](hermes-stt-enhance/README.md).** Start there for install, configuration, and the [privacy and data flow](hermes-stt-enhance/README.md#privacy-and-data-flow) notes. Audio stays local; with post-processing on (the default) the transcript goes to the LLM you configure, which may be remote.

```bash
hermes plugins install dmmeteo/hermes-stt-enhance#hermes-stt-enhance
```

## Repository layout

| Path | What |
|---|---|
| `hermes-stt-enhance/` | the plugin Hermes installs (`plugin.yaml`, code, README) |
| `tests/` | pytest suite; stubs the Hermes APIs, no Hermes checkout needed |
| `scripts/plugin_runtime.py` | inspect or pre-seed the Parakeet dependency runtime |
| `scripts/migrate_config.py` | rename a 0.4.x `local-llm-polished` profile config to 0.5.0 |
| `docs/` | dependency model and a Hermes core proposal |

```bash
python -m pytest -q
```

## License

MIT
