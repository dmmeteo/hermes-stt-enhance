# Next epics

The only outcome checklist for this repository. Statuses describe this public `main` revision (plugin 0.5.1). Implemented code is not runtime verification or user acceptance. An item is checked only after its completion condition is verified. Candidates are not selected and are not permission to implement.

## Delivered baseline

This revision implements the provider described in the [plugin README](../hermes-stt-enhance/README.md): faster-whisper and Parakeet backends, Parakeet chunking, the optional bounded cleanup pass with raw-transcript fallback, skill-sourced instructions, the dependency runtime outside the Hermes venv, and fail-closed refusal of 0.4.x names with `scripts/migrate_config.py`, and the fail-closed Parakeet install gate. This pass recorded no new runtime or user acceptance for it.

## Outcomes

- [ ] **Parakeet runtime install fails closed on the install policy.**
  - Status: implemented in 0.5.1 (`installs_allowed()` in `hermes-stt-enhance/runtime_deps.py`, covered by `tests/test_runtime_deps.py`). No runtime or user acceptance against a live Hermes is recorded.
  - Done when: with `security.allow_lazy_installs` unset, false, non-boolean or unreadable, or with `HERMES_DISABLE_LAZY_INSTALLS=1`, the first Parakeet use installs nothing and the user gets an error envelope or an unavailable provider. With an explicit `true` in the user's config, provisioning works.
  - Fails if: any setting other than an explicit `true` installs packages from PyPI.
  - Reference: [Gating](../docs/dependency-model.md#gating).
- [ ] **Install by catalog name.**
  - Status: waiting on the external Hermes plugin catalog entry. The README says the name works only after that entry lands.
  - Done when: `hermes plugins install hermes-stt-enhance` installs the reviewed, SHA-pinned release.
  - Fails if: the name installs an unpinned revision or a revision other than the reviewed one.
  - Reference: [Install](../hermes-stt-enhance/README.md#install).
- [ ] **Candidate, not selected: measured cleanup quality.**
  - Status: unselected. The README claims no word-error-rate improvement and records no benchmark.
  - Done when: an owner-approved reference set reports raw versus cleaned error rates per backend.
  - Fails if: a quality claim appears in user docs without that measurement.
  - Reference: [What the cleanup pass can and cannot do](../hermes-stt-enhance/README.md#what-the-cleanup-pass-can-and-cannot-do).
- [ ] **Candidate, not selected: plugin dependency lifecycle in Hermes core.**
  - Status: a proposal to Hermes, outside this repository. Nothing here depends on it.
  - Done when: Hermes ships a plugin dependency API and an owner decides to move `hermes-stt-enhance/runtime_deps.py` onto it.
  - Fails if: the move puts plugin packages back into the Hermes venv or lets them shadow Hermes modules.
  - Reference: [docs/hermes-core-proposal.md](../docs/hermes-core-proposal.md).
