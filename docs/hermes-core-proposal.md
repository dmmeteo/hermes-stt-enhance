# Proposal: a dependency lifecycle for Hermes plugins

**Status:** proposal, written against Hermes Agent 0.20.1 (`hermes_cli/plugins.py`
manifest v2, `tools/lazy_deps.py`).
**Scope:** small, additive, no behaviour change for plugins that do not opt in.

## The gap

Hermes has three dependency mechanisms and none of them covers a plugin's
optional runtime dependencies:

| Mechanism | What it does | Why it does not cover this |
|---|---|---|
| `plugin.yaml: python_dependencies` (#64165) | Validated, printed at install, warned about at load | Explicitly never installs. `_warn_python_dependencies` says so; the isolation design was deferred. |
| `tools/lazy_deps.py` | Installs allowlisted features into the core venv on first use | Allowlist is core-owned; plugins cannot add entries. `install_specs` accepts arbitrary specs but installs into the **core venv**, where a rebuild wipes them. |
| `hermes update` → `refresh_active_features()` | Reinstalls lazily-installed features after an update | Decides what to repair by what is **currently installed**, so it repairs nothing after a venv rebuild — the one case that needs repairing. Memory providers get a bespoke rescue path (`_refresh_active_memory_provider_dependencies`, #53272/#70636); no other plugin family does. |

The result is a silent failure with a misleading message. A plugin STT provider
whose optional package was wiped reports `is_available() == False`, and
`transcription_tools` turns that into:

```
STT plugin '<name>' is not available — check that its required credentials /
dependencies are configured.
```

The user's credentials and config are fine; a package is missing. Nothing in
the log says so, and `hermes update` will not fix it on the next run either.

That the memory-provider path needed a hand-written rescue (twice: mem0ai, then
hindsight-embed) is the signal that this belongs in the plugin API rather than
in per-family patches.

## What this plugin does instead

`hermes-stt-enhance` implements a plugin-owned runtime: dependencies live in a
content-addressed directory outside the venv, appended to `sys.path`, staged
and verified on install, with rollback to the previous runtime and one shared
copy per machine. See [dependency-model.md](dependency-model.md);
`hermes-stt-enhance/runtime_deps.py` is ~450 lines including the lifecycle.

It works, but every plugin with a compiled or heavyweight optional dependency
would have to write the same 450 lines, and each copy would get the ABI
handling, the staging, and the append-only precedence subtly differently.

## Proposal

### 1. Make `python_dependencies` installable, per plugin, off the venv

Extend the manifest with an optional runtime block:

```yaml
python_dependencies:
  - 'sherpa-onnx==1.13.4'
  - 'numpy==2.4.3'

python_runtime:
  mode: isolated        # isolated (default when the block is present) | venv | declare-only
  optional: true        # absent deps are not an error; the plugin probes and reports
  install: on-demand    # on-demand | on-install | never
```

`mode: declare-only` is today's behaviour and stays the default when the block
is absent, so no existing plugin changes.

### 2. Give `PluginContext` one method

```python
runtime = ctx.ensure_python_runtime()      # provisions if needed, activates, returns a handle
runtime.status()                           # satisfied / source / path / detail — never installs
```

Hermes already owns every piece this needs: the ABI stamping and wipe logic in
`lazy_deps._ensure_target_ready`, the append-only activation in
`_activate_target_on_syspath`, the uv→pip ladder in `_venv_pip_install`, spec
hygiene in `_spec_is_safe`, and the gate in `_allow_lazy_installs`. The
proposal is mostly about *where the packages land* and *who may ask*:

- target `<hermes home>/plugin-runtimes/<plugin id>/<abi>-<lock digest>/`,
  resolved above `profiles/<name>/` so profiles share one copy;
- install a clean directory per lock rather than `--target --upgrade`
  ([pypa/pip#13763](https://github.com/pypa/pip/issues/13763) — in-place
  upgrades leave two `dist-info` dirs for one package);
- keep the previous runtime, and fall back to it when an upgrade cannot reach
  the index, so an update can never take a working feature away;
- honour `security.allow_lazy_installs` exactly as today.

Allowing a plugin to name its own specs is a real trust change, but a smaller
one than it looks: `install_specs` already accepts arbitrary validated specs
from plugin manifests (memory providers), and a plugin that can register a
provider can already run arbitrary code. The gain is that its packages land
somewhere append-only, where they cannot shadow or downgrade core.

### 3. Repair on update, without guessing from what is installed

`hermes update` should ask each **enabled** plugin with a runtime to
re-provision, rather than inferring the work list from what survived:

```python
for manifest in enabled_plugins_with_python_runtime():
    ensure_python_runtime(manifest, prompt=False)   # never raises; reports per plugin
```

Because the runtimes live outside the venv, this is a fast no-op in the common
case — it only does work after a Python upgrade or a lock change. That is the
property `refresh_active_features` cannot have while it keys off installed
packages.

### 4. Surface it

- `hermes plugins doctor` — report each declared runtime: satisfied, where it
  resolves from, the exact provisioning command when it is not.
- `hermes doctor` — one line per plugin runtime, so a wiped or ABI-stale
  runtime is visible without reading logs.
- When a provider's `is_available()` is False, log the reason the provider
  gives. Today's dispatch discards it and prints a fixed sentence; a plugin
  that knows *why* it is unavailable cannot say so through that path.

### 5. Smallest useful subset

If only one thing lands, make it **(4)**: an unavailable provider that can
explain itself turns a silent outage into a one-line diagnosis. **(3)** is the
next-cheapest and removes the class of bug entirely for plugins that already do
their own provisioning.

## Compatibility

Every part is additive. Plugins without `python_runtime` behave exactly as they
do now, including the existing warn-only `python_dependencies`. A plugin that
adopts the API keeps working on older Hermes builds by falling back to its own
implementation — which is what `runtime_deps.py` here would do:

```python
try:
    runtime = ctx.ensure_python_runtime()
except AttributeError:
    runtime = plugin_owned_runtime()      # today's code path
```
