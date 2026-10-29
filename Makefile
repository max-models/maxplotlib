# The documentation site (docs/, Astro + Starlight).
# Notebook tool: starlight-notebooks, from @max-models/starlight-theme (docs/node_modules).
PYTHON ?= .venv/bin/python
DOCS_ENV = DOCS_PYTHON=$(abspath $(PYTHON))
NOTEBOOKS = cd docs && $(DOCS_ENV) npx starlight-notebooks

.PHONY: help docs-install docs-notebooks docs-dev docs-build docs-preview docs-clean

help:  ## list the targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

docs-install:  ## Python docs extra and the npm packages
	$(PYTHON) -m pip install -e ".[docs]"
	cd docs && npm ci

docs-notebooks:  ## execute the changed tutorials and convert all of them (FORCE=1 reruns all)
	$(NOTEBOOKS) execute $(if $(FORCE),--force,)
	$(NOTEBOOKS) convert

docs-dev:  ## live preview at http://localhost:4321/maxplotlib/
	$(NOTEBOOKS) convert
	cd docs && $(DOCS_ENV) DOCS_VALIDATE_LINKS=false npm run dev

docs-build:  ## the static site in docs/dist, as in CI
	$(NOTEBOOKS) convert
	cd docs && npm run check:mdx && $(DOCS_ENV) npm run build

docs-preview:  ## serve docs/dist
	cd docs && npm run preview

docs-clean:  ## remove generated pages, assets and build output
	$(NOTEBOOKS) clean
	rm -rf docs/dist docs/.astro
