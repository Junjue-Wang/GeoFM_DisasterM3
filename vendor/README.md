# vendor/ — offline install of `ever`

`ever` (Earth Vision Representation Library, https://gitee.com/zhuozheng/ever) is the
segmentation framework whose registry the data loaders in `code/data/` depend on. It is
**not distributed on PyPI**, so it is bundled here for offline installation.

## Contents
- `ever-2.0.0-py3-none-any.whl` — prebuilt, platform-independent wheel (pure Python). **Use this.**
- `ever_src/` — the exact source it was built from (`setup.py` + `ever/` package), in case you
  need to rebuild: `pip wheel --no-deps ./ever_src -w .`

## Install (offline)
`ever`'s own dependencies are ordinary PyPI packages and are pinned in `../requirements.txt`,
so install them first, then install `ever` from the wheel with `--no-deps`:

```bash
pip install -r ../requirements.txt                            # ever's deps (albumentations, scikit-image, ...)
pip install --no-index --no-deps ever-2.0.0-py3-none-any.whl  # ever itself, offline
```

## Fully air-gapped reviewer
If the review machine has no PyPI access at all, first mirror the PyPI wheels onto a machine
that does, then copy them over:

```bash
# on a networked machine (same python 3.12 / platform):
pip download -r requirements.txt -d wheelhouse/
# copy wheelhouse/ to the air-gapped machine, then:
pip install --no-index --find-links wheelhouse -r requirements.txt
pip install --no-index --no-deps vendor/ever-2.0.0-py3-none-any.whl
```

## Verified
The wheel was test-installed with `pip install --no-index --no-deps` into a clean target and
`import ever`, `import ever.api.data`, `import ever.interface` (the submodules the inference
code uses) all succeed. Version reported: `ever 2.0.0`.
