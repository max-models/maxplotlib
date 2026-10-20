// @ts-check
// The shared max-models docs setup (@max-models/starlight-theme): served at
// https://max-models.github.io/maxplotlib/, with math, the API reference and a link check.
import { defineDocs } from '@max-models/starlight-theme';

// The tutorial pages of the old Sphinx site (tutorials/tutorial_01.html, ...)
const tutorials = [
	'tutorial_01',
	'tutorial_02',
	'tutorial_03',
	'tutorial_04',
	'tutorial_05',
	'tutorial_06',
	'tutorial_07_tikz',
	'tutorial_08_plotly',
	'tutorial_09_plotext',
	'tutorial_10_matplotlib_nxm_spacing',
	'tutorial_11_gantt_charts',
	'tutorial_12_flame_charts',
	'tutorial_13_advanced_matplotlib',
	'tutorial_14_axis_and_layout_controls',
	'tutorial_15_tikzfigure_subplots',
	'tutorial_16_plotext_advanced',
	'tutorial_17_xarray',
];

export default defineDocs({
	repo: 'maxplotlib',
	title: 'maxplotlib',
	description: 'A reproducible plotting module with Matplotlib, Plotly, plotext and TikZ backends.',
	pythonPackage: 'maxplotlib',
	nav: [
		{ label: 'Getting started', link: '/getting-started/installation/' },
		{ label: 'Tutorials', link: '/tutorials/' },
		{ label: 'Guides', link: '/guides/matplotlib-import/' },
		{ label: 'Development', link: '/development/docs/' },
		{ label: 'API', link: '/api/maxplotlib/', section: '/api/' },
	],
	sidebar: [
		{
			label: 'Getting started',
			items: [
				{ label: 'Installation', slug: 'getting-started/installation' },
				{ label: 'Quickstart', slug: 'getting-started/quickstart' },
			],
		},
		{ label: 'Tutorials', items: [{ autogenerate: { directory: 'tutorials' } }] },
		{ label: 'Guides', items: [{ autogenerate: { directory: 'guides' } }] },
		{ label: 'Development', items: [{ label: 'Building the docs', slug: 'development/docs' }] },
	],
	astro: {
		redirects: Object.fromEntries(
			tutorials.map((name) => [`/tutorials/${name}.html`, `/tutorials/${name.replaceAll('_', '-')}/`])
		),
	},
});
