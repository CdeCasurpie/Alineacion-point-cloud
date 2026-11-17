# Explicit-view manual alignment

`blockalign.manual` is a separate, no-GPS path. Existing automatic manifests and
`Workspace.plan_add()` retain their contracts and behavior.

The local Barranco UI/control plane lives at
`/home/exitoso/home/barranco-roi`. See its README for HTTP routes, deployment and
input preparation. It snapshots both packages and the worker for each request.

## Numerical interface

- `audit(manifest, image_ids)`: sparse/database/image identities, SIFT shape and
  coordinates, immutable-WAL requirements and SHA256. No geography.
- `propose(source_views, target_views, selected, source_world, target_world, ...)`:
  bounded distance/direction suggestions; never replaces query selections.
- `measure(source, target, selection, scratch, policy, ...)`: native matching of
  explicit pairs and localization through the existing `pairwise.localize()`.
  Returns an outcome for every selected view, including failures. No per-edge
  minimum is imposed before joint evidence is available.
- `normalize(outcome, anchor_matrix, direction, anchor)`: both localization
  directions become incoming-local to fixed-world pose constraints.
- `solve(outcomes, prior, policy)`: robust seven-DoF fit using centers and
  orientations, capped anchor/repeated-view/quality weights, geometric
  degeneracy checks and reserved-view validation. All anchors are constants.
  Three or more anchors also produce leave-one-anchor-out sensitivity diagnostics.
- `constrained_ba(...)`: bounded sparse reprojection BA of incoming poses/points,
  fixed intrinsics and fixed anchor cameras. Uses only explicit fitting cross
  observations; reserves both incoming and reverse-query anchor views.
  Publishes a derived incoming model in map_world, not a merged anchor component.
  The caller must independently re-localize reserved views before accepting it.

Manual manifests need only `sparse.model_uri`, `features.database_uri` and
`images.root_uri`. These numerical functions run in the external supervised
worker, not an HTTP request. The caller pins geometry/placement revisions,
selection roles, explicit pairs, policy and engine identity.

`alignexp.binary.cameras()` now handles supported standard COLMAP camera models,
not only SIMPLE_RADIAL. The optional `selected` argument to
`colmap_adapter.feature_database()` creates a fresh 3.13 database containing only
approved pair endpoints. The default automatic path is unchanged.

## Tests

```bash
PATH=/data/barranco/runtime/alignment313/bin:$PATH PYTHONPATH=src \
  /data/barranco/runtime/alignment313/bin/python -B -m pytest tests -q
```

`tests/test_manual.py` exercises real native matching, forward/reverse PnP,
partial multianchor evidence, joint Sim(3), BA and unchanged source hashes using
synthetic inputs only. No production reconstruction is changed.
