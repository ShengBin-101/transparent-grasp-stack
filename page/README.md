# Project page

The source of the project page, https://shengbin-101.github.io/transparent-grasp-stack/. A GitHub Actions
workflow (`.github/workflows/pages.yml`) publishes this folder as it is: plain HTML, CSS and JavaScript with no
build step and no libraries.

Preview it locally:

```bash
cd page && python3 -m http.server 8000    # then open http://localhost:8000/
```

Python's server does not answer byte-range requests, which Safari needs for video, so check Safari on the
published page.

| Path | What it is |
|---|---|
| `index.html` | the page |
| `static/css/index.css`, `static/js/index.js` | styles (light and dark) and behaviour: videos play only while on screen, with a pause button, and not at all with reduced motion; the depth slider; chart tooltips; copy BibTeX |
| `static/videos/` | H.264 clips without audio, each with a poster frame: the exploratory table clear (`teaser`), and the plastic bowl under the three trial conditions (`bowl`) |
| `static/images/` | a still of the point clouds (`clouds_still`), the radar figures (`radar_*`), the first capture of the table clear (colour, the camera's depth and the repaired depth as height above the table, the segmenter's mask, the grasps), the setup photo, the radar rig, the social preview |
| `static/viewer/point_clouds.html` | the offline demo's point clouds in an interactive viewer (three.js r128, MIT, inlined), loaded into the page on request; `#2` opens the second scene |
| `radar.html` | the radar study of the project's first half |

## Credit and licence

The page's design and code adapt the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template)
by Eliahu Horwitz, itself adapted from the [Nerfies](https://nerfies.github.io) page, and are licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) like it. The rest of the repository is under its own
[licence](../LICENSE).
