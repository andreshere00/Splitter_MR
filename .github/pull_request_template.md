## Summary

<!-- 1–3 bullets: what changed and why. -->

-

## Type of change

- [ ] Feature
- [ ] Bug fix
- [ ] Breaking change
- [ ] Documentation
- [ ] Developer / CI

## Test plan

- [ ] `poe format` after Python changes
- [ ] Targeted tests for the changed code
- [ ] `poe test` when full verification is needed (coverage ≥ 70%)

## Documentation

- [ ] `README.md` updated if the public API, features, or examples changed
- [ ] `CHANGELOG.md` updated (`Unreleased` if no version is assigned yet)
- [ ] Did not hand-edit `docs/index.md` or `docs/CHANGELOG.md`

## Checklist

- [ ] Follows existing reader / splitter / model / embedding patterns
- [ ] Optional dependencies stay behind lazy imports
- [ ] New public types are exported from the relevant package `__init__.py`
- [ ] Tests follow Arrange-Act-Assert and `test_{method}_{state}_{expected}`
