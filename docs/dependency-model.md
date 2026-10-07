# Dependency model

How this plugin keeps `sherpa-onnx` and `numpy` working across Hermes updates,
and why it does not put them in Hermes' venv.

## The failure it is designed around

`sherpa-onnx` and `numpy` are not Hermes core dependencies. Hermes installs
packages like these *lazily into its own venv* (`tools/lazy_deps.py`), and a
Hermes update can recreate that venv from `pyproject.toml` — which wipes
everything that was installed lazily.

Hermes has a repair pass for this (`lazy_deps.refresh_active_features`, run by
`hermes update`), but it can only heal features whose **anchor package is still
present**: it decides what to reinstall by checking what is installed. After a
rebuild nothing is installed, so nothing looks active, so nothing is repaired.
The packages stay gone, and because Hermes returns

```
STT plugin 'stt_enhance' is not available — check that its required
credentials / dependencies are configured.
```

*instead of* calling the provider when `is_available()` is False, the failure
surfaces as a generic misconfiguration message rather than "a package is
missing". That is the exact outage this model exists to prevent.

## The model

The plugin owns a runtime directory **outside** Hermes' venv:

```
~/.hermes/plugin-runtimes/hermes-stt-enhance/cpython-311-linux-x86_64-d258e225fdb6/
├── sherpa_onnx/
├── numpy/
└── runtime.json          ← written last; the completion marker
```

The directory name is content-addressed: `<interpreter ABI>-<platform>-<lock
digest>`. Nearly every lifecycle property falls out of that one decision.

| Event | What happens |
|---|---|
| Fresh install | First voice message provisions the runtime, then transcribes. One slow message, no setup step. |
| **Hermes venv rebuild** | Nothing. The directory is not in the venv, so the next message activates it and transcribes — no network, no install. |
| Dependency upgrade | New lock digest → new directory → clean install. The old one stays. |
| Python upgrade | New ABI tag → new directory. Wheels compiled for the old interpreter are never imported by the new one. |
| Failed/offline upgrade | The previous runtime still verifies, so it is activated instead. Transcription keeps working; the upgrade simply has not been applied. |
| Rollback | Pin the old versions back: that digest's directory is still on disk, so it is a no-network reactivation. |
| Several profiles | One shared install — the root resolves above `profiles/<name>/`. |
| Disk creep | After a successful install, superseded runtimes beyond the newest one are pruned. |

### Why a directory per lock, not `pip install --target --upgrade`

`pip install --target` does not remove the version it replaces; upgrading in
place leaves two `dist-info` directories for one package, which makes
`importlib.metadata` ambiguous about what is installed
([pypa/pip#13763](https://github.com/pypa/pip/issues/13763)). A clean directory
per lock avoids that entirely and makes "which versions are in here" a question
with one answer.

### Why not a nested venv

A venv would add an interpreter indirection, its own bootstrap cost, and a
second copy of every shared package, and it would still need the same ABI and
integrity handling. `--target` plus `sys.path` activation gives the same
isolation for this problem — a handful of packages loaded into the Hermes
process — without a second environment to keep in sync.

### Precedence: the runtime can never break Hermes

The directory is **appended** to `sys.path`, never prepended, using the same
approach as Hermes' own durable lazy-install target
(`tools.lazy_deps._activate_target_on_syspath`): `site.addsitedir` so `.pth`
files are honoured, then the ordering is re-enforced. Anything Hermes core
ships wins every name collision. The worst a bad install here can do is fail to
import and report the backend unavailable.

If the packages are *already* importable — because Hermes core happens to ship
them, or an operator installed them into the venv — the plugin uses those and
provisions nothing.

### Integrity

- Exact pins, never ranges, so the digest means something.
- Installs stage into a temporary directory beside the target and land with a
  single rename; `runtime.json` is written last, so an interrupted or offline
  install leaves nothing a later run would trust.
- After install, the versions actually resolved are compared against the lock;
  a mismatch is refused rather than used.
- Before use, the recorded lock, schema, interpreter, and on-disk versions are
  all re-checked. A drifted or tampered runtime is rebuilt, not trusted.
- Only PyPI names with exact versions are ever passed to the installer — no
  URLs, paths, or indexes from config.

Hash-pinned installs (`--require-hashes`) are the natural next step and are not
implemented: it needs a generated lock file with per-wheel hashes for every
platform the plugin supports.

### Gating

Runtime installs stay under the user's control:

- Hermes-wide, provisioning needs `security.allow_lazy_installs: true` set
  in the user's `config.yaml` and in Hermes' effective config
  (`hermes_cli.config.load_config_readonly()`). Hermes' built-in default of
  `true` does not count. False, unset, non-boolean (`"false"`, `1`), or an
  unreadable or malformed config file blocks it, and so does Hermes'
  sealed-environment switch `HERMES_DISABLE_LAZY_INSTALLS=1`. A `true`
  pinned only by a managed config does not count either.
- `stt.stt_enhance.parakeet.auto_install_deps: false` blocks it for this
  plugin only.

Either way an already-provisioned runtime is still used — freezing installs
disables *installing*, not *working*. When provisioning is blocked and nothing
is on disk, the error names the exact command, in both the log and the error
envelope the user sees.

## Operating it

```bash
# Everything below is optional — first use provisions itself.
P=~/.hermes/hermes-agent/venv/bin/python      # the interpreter Hermes runs

$P scripts/plugin_runtime.py --status         # what is installed, where it resolves from
$P scripts/plugin_runtime.py --install        # provision now (image build, pre-seed)
$P scripts/plugin_runtime.py --verify         # activate + report each dependency's origin
$P scripts/plugin_runtime.py --prune --keep 1 # reclaim superseded runtimes
$P scripts/plugin_runtime.py --status --json  # machine-readable, for health checks
```

`--verify` never installs, so it is the honest check after an air-gapped
pre-seed or an image build.

### Air-gapped installs

Provision the directory by hand with a local wheel source — the path is the one
`--status` prints:

```bash
$P -m pip install --target '<runtime dir>' --no-index --find-links /media/wheels \
   'sherpa-onnx==1.13.4' 'numpy==2.4.3'
```

Then `--verify` to confirm, and set `auto_install_deps: false` if the machine
should never attempt an install.

### Moving the runtime

`HERMES_STT_ENHANCE_RUNTIME_ROOT` overrides the location — useful when
`~/.hermes` is small, or when a read-only image wants the runtime on a data
volume.
