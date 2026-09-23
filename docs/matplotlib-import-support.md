# Matplotlib import support and roadmap

`Canvas.from_matplotlib(source, strict=False, **canvas_kwargs)` accepts an
in-memory Matplotlib figure, axes, or rectangular axes array. This checklist
covers the intended import surface. It is a roadmap, not a promise that every
item already works on every rendering backend.

Checked items are implemented within their stated scope. Unchecked items are
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
- [ ] Structured import report with artist identities, severity, and fallbacks.
- [ ] Complete detection of unsupported style properties in strict mode.
- [ ] Version compatibility matrix and fixtures across supported Matplotlib versions.
- [ ] Optional serialized figure input with an explicit trust boundary.

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
- [ ] Shared-axis relationships and linked limits after import.
- [ ] Preserve omitted/empty grid cells without drawing extra empty axes.
- [ ] Spanning cells, subplot mosaics, nested GridSpec, and subfigures.
- [ ] Grid width/height ratios, margins, spacing, constrained/tight layout.
- [ ] Arbitrary axes rectangles, overlapping axes, inset axes, inset connectors.
- [ ] Secondary axes with forward/inverse coordinate functions.
- [ ] Figure backgrounds, frame styling, and figure-level text/patches/images.
- [ ] Figure title/shared-label typography and placement.
- [ ] Figure-level legends and shared colorbars.

Irregular grids currently warn and fall back to a single row. A 2D input array
defines primary-axis slots; slots occupied only by a twin are not compacted.

## Lines, markers, and collections

- [x] Numeric 2D lines, markers, NaN gaps, line/marker colors and widths.
- [x] Step draw styles in imported line entries (Matplotlib rendering).
- [x] Horizontal/vertical reference lines with fractional extents (Matplotlib).
- [x] Plain line collections, including `hlines`/`vlines`, as individual lines.
- [x] Standard scatter marker shapes, sizes, scalar colors, and per-point colors.
- [x] Scatter colormaps and normalization copied for Matplotlib rendering.
- [ ] Preserve date/time, categorical, quantity/unit converters and formatters.
- [ ] Custom dash sequences, cap/join styles, markevery, and gap colors.
- [ ] Infinite `axline` semantics, event plots, stem containers.
- [ ] Scalar-mapped line collections, offset collections, per-point transforms.
- [ ] Custom scatter paths and hollow markers on every backend.
- [ ] Match scatter area/size semantics and normalization on every backend.
- [ ] Preserve ordering between artist types at equal z-order.

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
- [ ] Error limits/arrows, subsampled errors, and independently styled components.
- [ ] Restore semantic fill boundaries/where masks when recoverable.
- [ ] Spans in blended coordinates; general rectangles, circles, ellipses, wedges.
- [ ] Compound polygons, holes, curved paths, arbitrary PathPatch geometry.
- [ ] Box plots, violin plots, pie/donut charts, hist2d, hexbin.
- [ ] Preserve statistical groupings when containers/metadata retain them.

## Images, fields, and colorbars

- [x] `imshow` arrays, extent, origin, colormap, normalization, interpolation
  copied into entries (Matplotlib rendering).
- [ ] Match image extent/origin, RGB(A), masks, alpha, and interpolation on all backends.
- [ ] Nonlinear normalization, clim, under/over/bad colors on all backends.
- [ ] `pcolor`, `pcolormesh`, nonuniform grids, QuadMesh, triangular meshes.
- [ ] Contours, filled contours, levels, labels, and contour topology.
- [ ] Quiver, barbs, streamplots, vector-field keys.
- [ ] Axes colorbars tied to the correct image/scatter/mesh mappable.
- [ ] Colorbar orientation, label, ticks, limits, extend, and normalization.
- [ ] Multiple/shared colorbars and explicit colorbar axes.

## Text and annotations

- [x] Data-coordinate text, font family/size/weight/style, color, alignment,
  rotation (backend support varies).
- [x] Data-coordinate annotations with optional arrows and independent copied
  arrow properties; references to other patches are explicitly rejected.
- [x] Center title and x/y labels with basic typography and label padding.
- [ ] Text boxes, multiline spacing, math/TeX fidelity, wrapping, clipping.
- [ ] Axes/figure fractions, point/pixel offsets, blended and callable coordinates.
- [ ] Annotation arrows with full backend parity and artist-relative coordinates.
- [ ] Left/right titles, custom title/label positions, offset text.
- [ ] AnnotationBbox, OffsetBox, tables, and other composite text artists.

## Axes, ticks, grids, and legends

- [x] Limits including reversed limits; default linear/log scales.
- [x] Explicit major tick locations/labels with FixedLocator.
- [x] Basic grid visibility, axes visibility, background, aspect, axisbelow.
- [x] Basic per-axes legend visibility and artist labels.
- [ ] Non-default log bases; symlog, logit, asinh, function/custom scales.
- [ ] Automatic/fixed minor ticks, locator/formatter configuration and units.
- [ ] Tick placement, direction, length, width, color, rotation, and font styling.
- [ ] Per-axis major/minor grid visibility and line styling.
- [ ] Spine visibility, colors, widths, bounds, and positions.
- [ ] Autoscale flags, sticky edges, margins, adjustable/anchor/box aspect.
- [ ] Legend order, renamed labels, proxy handles, multiple legends, grouping.
- [ ] Legend location/anchor, columns, title, typography, frame and spacing.

## Transforms, metadata, and advanced axes

- [ ] General affine/nonlinear/blended transforms, coordinate rebinding.
- [ ] Clip paths/boxes, path effects, rasterization, sketch settings, filters.
- [ ] Hidden artists retained as hidden editable entries (currently skipped).
- [ ] Artist IDs, URLs, picking, metadata, and accessibility descriptions.
- [ ] Polar, geographic/custom projections, axisartist and parasite axes.
- [ ] 3D lines/scatter, surfaces, wireframes, collections, camera/projection.
- [ ] Animations, widgets, callbacks, and interactive state (separate adapters).
- [ ] Optional raster fallback for unsupported artists with explicit loss reporting.

## Validation and next priorities

- [x] Tests for all three input forms, source independence, strict/warning modes.
- [x] Matplotlib reconstruction tests for supported geometry and axis settings.
- [x] Plotly smoke/geometry tests for representative supported imports.
- [ ] Image comparisons with tolerances, plus export tests for all backends.
- [ ] Large figures, empty/masked data, performance and memory benchmarks.

Next priorities: colorbar ownership; layout/shared-axis fidelity; axis scales,
ticks and legends; image/mesh support; comprehensive diagnostics. Add semantic
entries when source metadata supports them, otherwise identify geometry-only
imports and backend limitations explicitly.

Reference APIs: [Matplotlib artists](https://matplotlib.org/stable/tutorials/artists.html),
[containers](https://matplotlib.org/stable/api/container_api.html), and
[annotations](https://matplotlib.org/stable/users/explain/text/annotations.html).
