# Matplotlib import support and roadmap

`Canvas.from_matplotlib(source, strict=False, **canvas_kwargs)` accepts an
in-memory Matplotlib figure, axes, or rectangular axes array. This checklist
covers the intended import surface. It is a roadmap, not a promise that every
item already works on every rendering backend.

Checked items are implemented within their stated scope. Unless explicitly
stated otherwise, fidelity checks below refer to the **Matplotlib backend**.
Complex built-in artists are retained as detached, editable native geometry;
these entries deliberately raise on backends that cannot represent them.
Portable entries continue to support the other renderers, with the limitations
listed below. Unchecked items are
pending or partial. Importing drawn geometry does not recover the original
samples, plotting call, callback, or statistical model. Matplotlib, Plotly,
TikZ, and plotext have different rendering capabilities; import success does
not guarantee that every backend can render the result identically.

## Inputs and ownership

- [x] Whole `Figure`, individual `Axes`, flat lists/arrays, and 2D lists/arrays.
- [x] Explicit 2D array order; infer ordinary grids from flat input.
- [x] Reject empty, ragged, non-axes, duplicate, and mixed-figure inputs.
- [x] Snapshot plot arrays and styles without reparenting source artists.
- [x] Canvas keyword overrides for figure size, DPI, and other canvas options.
- [x] Warnings for recognized unsupported content; strict mode raises.
- [x] Structured import report with artist identities, severity, and fallbacks.
- [ ] Complete detection of unsupported style properties in strict mode.
- [ ] Version compatibility matrix and fixtures across supported Matplotlib versions.
- [x] Optional serialized figure input with an explicit trust boundary.

PNG/JPEG input and reconstruction of original data from PDF/SVG are separate
features, not part of this object importer. Arbitrary custom Python artists
need an adapter API or an explicit raster fallback; universal semantic
conversion is not possible.

## Figure and layout

- [x] Ordinary rectangular subplot grids and individual axes.
- [x] Figure dimensions and export DPI defaults.
- [x] Figure title and shared axis label text.
- [x] One `twinx` and one `twiny` per primary subplot; import only selected axes.
- [ ] Full twin-axis spine positioning, multiple twins, and cross-backend parity.
  Matplotlib supports multiple twins and spine positions; backend parity is pending.
- [x] Shared-axis relationships and linked limits after import.
- [x] Preserve omitted/empty grid cells without drawing extra empty axes.
- [x] Spanning cells, subplot mosaics, nested GridSpec, and subfigures
  (subfigure rectangles/decorations are flattened into the destination figure).
- [x] Grid width/height ratios, margins, spacing, constrained/tight layout
  for ordinary/nested GridSpec; subfigures retain their snapshot rectangles.
- [x] Arbitrary axes rectangles, overlapping axes, inset axes, inset-zoom
  connectors (`indicate_inset_zoom`/`indicate_inset`, Matplotlib 3.10+). The
  connector spans two Axes and recomputes its geometry from live limits on
  every draw, so it is rebuilt against the reconstructed parent/inset pair
  rather than snapshotted. Matplotlib < 3.10's tuple-returning API still falls
  back to native geometry.
- [x] Secondary axes with forward/inverse coordinate functions.
- [x] Figure backgrounds, frame styling, and figure-level text/patches/images.
- [x] Figure title/shared-label typography and placement.
- [x] Figure-level legends and shared colorbars.

Irregular grids use one row of addressable Canvas slots and retain the source
axes rectangles when rendered with Matplotlib. A 2D input array defines its own
layout; slots occupied only by a twin remain empty rather than drawing extra axes.
Ordinary/nested GridSpec layout engines reflow on resize. Subfigure layout-engine
reflow and full cross-backend layout parity remain pending.

## Lines, markers, and collections

- [x] Numeric 2D lines, markers, NaN gaps, line/marker colors and widths.
- [x] Step draw styles in imported line entries (Matplotlib rendering).
- [x] Horizontal/vertical reference lines with fractional extents (Matplotlib).
- [x] Plain line collections, including `hlines`/`vlines`, as individual lines.
- [x] Standard scatter marker shapes, sizes, scalar colors, and per-point colors.
- [x] Scatter colormaps and normalization copied for Matplotlib rendering.
- [x] Preserve date/time, categorical, quantity/unit converters and formatters.
- [x] Custom dash sequences, cap/join styles, markevery, and gap colors.
- [x] Infinite `axline` semantics, event plots, stem containers.
- [x] Scalar-mapped line collections, offset collections, per-point transforms.
- [ ] Custom scatter paths and hollow markers on every backend.
- [ ] Match scatter area/size semantics and normalization on every backend.
- [x] Preserve ordering between artist types at equal z-order.

## Bars, errors, fills, and statistical plots

- [x] Vertical/horizontal bars, positions, dimensions, baselines, and styling.
- [x] Stacked/grouped bars as their resolved rectangles.
- [x] Histogram bars as geometry (original samples are unavailable).
- [x] Error-bar containers with data lines: symmetric/asymmetric x/y errors,
  caps, colors, widths, and labels; avoid duplicated component artists.
- [x] Error-only plots and bar errors as line/cap geometry, without inventing
  asymmetric centers that the source no longer retains.
- [x] Simple data-coordinate polygon collections, including `fill_between`,
  `fill_betweenx`, and stackplot regions, as filled polygon geometry.
- [x] Data-coordinate polygon patches and stairs values/edges/baseline.
- [x] Error limits/arrows, subsampled errors, and independently styled components.
- [ ] Restore semantic fill boundaries/where masks when recoverable.
  Drawn paths and retained container metadata are preserved; Matplotlib usually
  does not retain the original where-mask or samples.
- [x] Spans in blended coordinates; general rectangles, circles, ellipses, wedges.
- [x] Compound polygons, holes, curved paths, arbitrary PathPatch geometry.
- [x] Box plots, violin plots, pie/donut charts, hist2d, hexbin.
- [x] Preserve statistical groupings when containers/metadata retain them:
  `subplot.import_groups` and entry `source_container_ids` retain bar, errorbar,
  and stem memberships, labels, orientation, and available data values.

## Images, fields, and colorbars

- [x] `imshow` arrays, extent, origin, colormap, normalization, interpolation
  copied into entries (Matplotlib rendering).
- [ ] Match image extent/origin, RGB(A), masks, alpha, and interpolation on all backends.
- [ ] Nonlinear normalization, clim, under/over/bad colors on all backends.
- [x] `pcolor`, `pcolormesh`, nonuniform grids, QuadMesh, triangular meshes.
- [x] Contours, filled contours, levels, labels, and contour topology.
- [x] Quiver, barbs, streamplots, vector-field keys.
- [x] Axes colorbars tied to the correct image/scatter/mesh mappable.
- [x] Colorbar orientation, label, ticks, limits, extend, and normalization.
- [x] Multiple/shared colorbars and explicit colorbar axes.

## Text and annotations

- [x] Data-coordinate text, font family/size/weight/style, color, alignment,
  rotation (backend support varies).
- [x] Data-coordinate annotations with optional arrows and independent copied
  arrow properties; references to other patches are explicitly rejected.
- [x] Center title and x/y labels with basic typography and label padding.
- [x] Text boxes, multiline spacing, math/TeX fidelity, wrapping, clipping
  through Matplotlib (TeX still requires the source environment’s TeX setup).
- [ ] Axes/figure fractions, point/pixel offsets, blended and callable coordinates.
- [ ] Annotation arrows with full backend parity and artist-relative coordinates.
- [x] Left/right titles, custom title/label positions, offset text.
- [x] AnnotationBbox, OffsetBox, tables, and other composite text artists.

## Axes, ticks, grids, and legends

- [x] Limits including reversed limits; default linear/log scales.
- [x] Explicit major tick locations/labels with FixedLocator.
- [x] Basic grid visibility, axes visibility, background, aspect, axisbelow.
- [x] Basic per-axes legend visibility and artist labels.
- [x] Non-default log bases; symlog, logit, asinh, function/custom scales.
- [x] Automatic/fixed minor ticks, locator/formatter configuration and units.
- [x] Tick placement, direction, length, width, color, rotation, and font styling.
- [x] Per-axis major/minor grid visibility and line styling.
- [x] Spine visibility, colors, widths, bounds, and positions.
- [x] Autoscale flags, sticky edges, margins, adjustable/anchor/box aspect.
- [x] Legend order, renamed labels, proxy handles, multiple legends, grouping.
- [x] Legend location/anchor, columns, title, typography, frame and spacing.

## Transforms, metadata, and advanced axes

- [x] General affine/nonlinear/blended transforms, coordinate rebinding.
- [x] Clip paths/boxes, path effects, rasterization, sketch settings, filters.
- [x] Hidden artists retained as hidden editable entries.
- [ ] Artist IDs, URLs, picking, metadata, and accessibility descriptions.
- [x] Polar, geographic/custom projections, axisartist and parasite axes via
  native axes snapshots; custom projection classes must remain available.
- [x] 3D lines/scatter, surfaces, wireframes, collections, camera/projection.
- [ ] Animations, widgets, callbacks, and interactive state (separate adapters).
  Static artist state is copied; arbitrary callback closures and GUI event loops
  cannot be reconstructed from a figure’s drawn geometry.
- [x] Optional raster fallback for unsupported artists with explicit loss reporting.

## Validation and next priorities

- [x] Tests for all three input forms, source independence, strict/warning modes.
- [x] Matplotlib reconstruction tests for supported geometry and axis settings.
- [x] Plotly smoke/geometry tests for representative supported imports.
- [x] Image comparisons with tolerances, plus representative portable export
  tests for Matplotlib PNG/SVG, Plotly HTML, plotext text, and TikZ source.
- [x] Large figures, empty/masked data, performance and memory benchmarks.

New import controls:

```python
canvas = Canvas.from_matplotlib(fig, strict=True)
report = canvas.import_report.to_dict()  # identities, severity, fallback, backends
canvas = Canvas.from_matplotlib("figure.pickle", trusted=True)
canvas = Canvas.from_matplotlib(fig, fallback="raster")
```

`trusted=True` is required before any pickle is read. Pickles can execute code;
load only files whose producer you trust and use a matching Matplotlib version.
`fallback="native"` (default) retains built-in native geometry with rebound
transforms. `fallback="skip"` warns (or raises in strict mode) for those artists.
`fallback="raster"` snapshots the selected figure content and records the loss of
editable data and vector geometry; it supports Matplotlib and Plotly. Callback
functions and custom scales remain Python objects, not portable serialization.
Strict mode checks import losses, not universal rendering parity.

Remaining priorities are cross-backend fidelity, subfigure reflow, portable
projection adapters, and live interaction adapters. Source metadata does not
usually retain original fill masks or box/violin samples; these are imported as
geometry without inventing lost statistical inputs.

Reference APIs: [Matplotlib artists](https://matplotlib.org/stable/tutorials/artists.html),
[containers](https://matplotlib.org/stable/api/container_api.html), and
[annotations](https://matplotlib.org/stable/users/explain/text/annotations.html).


Compatibility fixtures live in `test_matplotlib_import.py` and
`test_matplotlib_import_extended.py`. The CI matrix covers Matplotlib 3.8, 3.9,
3.10 and 3.11 on Python 3.11. Snapshots use version-sensitive Matplotlib state;
there is no cross-version pickle compatibility promise.

Run `python scripts/benchmark_matplotlib_import.py --points 100000 --panels 4`
for reproducible timing/allocation observations. A local Matplotlib 3.11.1 run
imported 400,000 points in 0.41 s with 8.75 MiB peak Python allocations and rendered
in 0.79 s. These are observations, not performance guarantees; native allocations
outside Python are excluded from the memory measurement.
