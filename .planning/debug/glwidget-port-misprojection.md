---
status: awaiting_human_verify
trigger: "BALLView renders embedded but ball-and-stick bonds draw as enormous mis-projected cylinders off the viewport edges after QGLWidget->QOpenGLWidget port"
created: 2026-05-14
updated: 2026-05-14
---

## Current Focus

hypothesis: CONFIRMED — HiDPI mismatch. GLRenderer::glViewport uploaded logical pixels (857x577) into a device-pixel QOpenGLWidget FBO (1714x1154 at dpr=2), so the scene rendered into only the bottom-left quarter of the framebuffer.
test: instrumentation printed active viewport vs FBO size at renderToBuffer time
expecting: viewport != FBO size — confirmed (857x577 viewport in 1714x1154 FBO)
next_action: user visual recheck

## Symptoms

expected: RQI tripeptide renders as recognizable ball-and-stick molecule embedded in Scene viewport
actual: bonds draw as enormous mis-projected green/cyan cylinders extending far off viewport edges; molecule unrecognizable
errors: "Cannot resize window. Size 0 x 0 is not supported" in startup log
reproduction: build BALL VIEW BALLView, run headless smoke, demo builds RQI peptide automatically
started: after QGLWidget->QOpenGLWidget port (GSD Phase 2)

## Eliminated

## Evidence

- timestamp: 2026-05-14
  checked: GLRenderer::renderToBuffer (glRenderer.C:547)
  found: renderToBuffer does NOT set its own viewport or projection — relies entirely on state set by setSize()/initPerspective()/updateCamera() earlier
  implication: whatever viewport+frustum setSize installs is what renderToBuffer uses

- timestamp: 2026-05-14
  checked: GLRenderer::setSize (glRenderer.C:2075), setProjection (2375)
  found: setSize sets width_/height_, x_scale_/y_scale_ from aspect ratio, glViewport(0,0,width_,height_), initPerspective->setProjection->glFrustum(left_..). left_=-2*x_scale_ etc.
  implication: viewport and frustum both derive from width_/height_. Internally consistent ONLY if width_/height_ match the real framebuffer pixel size.

- timestamp: 2026-05-14
  checked: Scene::resizeEvent (scene.C:558), RenderSetup::resize (renderSetup.C:152), Scene::initializeGLContext (scene.C:3376)
  found: renderers get Scene::width()/height() = LOGICAL pixels. QOpenGLWidget default FBO is DEVICE pixels (Retina 2x).
  implication: glViewport gets logical px into a device-px FBO -> viewport covers only bottom-left quarter on Retina

- timestamp: 2026-05-14
  checked: instrumented GLRenderer::renderToBuffer to print active GL_VIEWPORT + GL_FRAMEBUFFER_BINDING
  found: active viewport=0,0,857x577 while QOpenGLWidget FBO (binding=1) is device-pixel sized 1714x1154 (dpr=2)
  implication: ROOT CAUSE — GLRenderer renders into bottom-left quarter of the framebuffer; QOpenGLWidget composites the whole FBO -> molecule grossly mis-placed/mis-scaled

- timestamp: 2026-05-14
  checked: post-fix instrumentation
  found: logical=857x577 pixel_ratio=2 glViewport=1714x1154 — viewport now matches FBO exactly; frustum aspect 1.485 correct; 0 GL errors, 0 exceptions; 0x0 resize no longer produces NaN frustum
  implication: fix verified at log level; awaiting user visual confirmation

## Resolution

root_cause: |
  QGLWidget->QOpenGLWidget port. QOpenGLWidget's default framebuffer is
  device-pixel sized (Retina dpr=2 => 2x logical). GLRenderer::setSize() and
  setupStereo() upload glViewport() using width_/height_ which are LOGICAL
  pixels (from Scene::width()/height()). Result: the scene was rendered into
  only the bottom-left quarter of the device-pixel framebuffer, and
  QOpenGLWidget then composited the whole FBO — producing the grossly
  mis-projected/oversized geometry. A transient 0x0 resize also left a NaN
  frustum (harmless because a real resize follows, but cleaned up too).
fix: |
  Added GLRenderer::pixel_ratio_ (default 1.0) + setPixelRatio()/getPixelRatio().
  RenderSetup::resize() sets it from gl_target_->devicePixelRatioF() before
  calling renderer->setSize(). The pixel_ratio_ factor is applied ONLY at the
  three GL coordinate-space sites: the two glViewport() uploads (setSize,
  setupStereo) and gluPickMatrix() in pickObjects1() so picking still lines up.
  width_/height_ and all viewport-to-3D math stay logical, so mapViewportTo3D
  and getWidth/getHeight are unchanged. Added a width<1||height<1 guard in
  setSize() to skip the degenerate 0x0 resize. On non-HiDPI displays
  devicePixelRatioF()==1.0 so the change is a no-op — platform-independent.
verification: |
  Headless smoke test: final state logical=857x577 pixel_ratio=2
  glViewport=1714x1154 (matches FBO exactly), frustum aspect 1.485 correct
  and non-degenerate, 0 GL errors, 0 exceptions, demo peptide builds cleanly.
  Production binary (instrumentation removed) re-run: 0 errors/exceptions.
  Awaiting user visual confirmation that the RQI tripeptide now renders as a
  correctly-sized, centered ball-and-stick molecule.
files_changed:
  - include/BALL/VIEW/RENDERING/RENDERERS/glRenderer.h
  - source/VIEW/RENDERING/RENDERERS/glRenderer.C
  - source/VIEW/RENDERING/renderSetup.C
