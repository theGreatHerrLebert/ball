# Renderer Interface Boundary — making Phase 5 ("4b") a contained swap

**Status:** Draft · 2026-05-14
**Purpose:** Define the abstraction boundary so the Phase 5 rendering-pipeline
modernization (GL core profile **or** QRhi) is "implement one new backend +
register it", not "re-thread the whole widget again".
**Consumed by:** `/gsd-plan-phase 5`. Also informs the tail of Phase 2.

---

## The good news: the abstraction is ~80% already there

BALL already has two clean-ish interfaces. The Phase 5 swap is contained if
and only if we finish them — it is *not* a from-scratch design.

| Interface | File | State |
|-----------|------|-------|
| `Renderer` (abstract) | `include/BALL/VIEW/RENDERING/RENDERERS/renderer.h` | Good. `init / render_ / renderSphere_ / … / updateCamera / pickObjects / mapViewportTo3D`. **No Qt-GL types in its signatures.** Concrete impls: `GLRenderer`, `POVRenderer`, `STLRenderer`, `VRMLRenderer`, `XML3DRenderer`, `raytracingRenderer`, `tilingRenderer`. |
| `RenderTarget` (abstract) | `include/BALL/VIEW/RENDERING/renderTarget.h` | Good. `getBuffer / getFormat / releaseBuffer / init / resize / refresh / prepareRendering / doNotResize`. **No Qt-GL types either.** |
| `RenderSetup` (QThread) | `include/BALL/VIEW/RENDERING/renderSetup.h` | Mostly GL-agnostic — ties a `Renderer*` + `RenderTarget*`. **One leak:** `virtual makeCurrent()`. |
| `GLRenderWindow` | `glRenderWindow.h` | **The leak sink.** Multiply-inherits `RenderWindow` (CPU buffer) **and** `QOpenGLWidget` (Qt widget + GL context). Fuses three concerns. |

## What actually leaks (the 4 things Phase 5 would otherwise have to chase)

1. **`RenderSetup::makeCurrent()`** is GL-context-specific and is called from
   `renderSetup.C` (~13 sites) and `scene.C`. A QRhi backend has no
   "make current" — it has command buffers. This verb must become
   backend-neutral or move behind the surface.
2. **`scene.C` `dynamic_cast<GLRenderWindow*>` / `dynamic_cast<GLRenderer*>`**
   (~6 sites) and **`new GLRenderWindow(...)`** (main display + the deferred
   stereo paths). `Scene` should never name a concrete backend type.
3. **`GLRenderWindow` fuses surface + context + CPU buffer.** `QOpenGLWidget`,
   the GL context, *and* the `RenderWindow` pixel buffer are one class. A
   QRhi/GL-core surface has a different context model entirely.
4. **`Renderer`'s per-primitive `render*_()` virtuals are immediate-mode
   shaped** — `renderSphere_(const Sphere&)` assumes "issue a draw call per
   object now". A retained/batched backend (QRhi, GL-core VBOs) wants
   "give me all spheres, I'll build a buffer". This is the deepest leak and
   the one most worth fixing at the interface, not inside each backend.

## Target boundary

Three interfaces, each with **zero Qt-GL types in its signatures**. `Scene`
and `RenderSetup` talk only to these.

### 1. `RenderBackend` — keep `Renderer`, tighten the contract

Keep the `Renderer` base class; it is the right seam. Phase 5's GL-core or
QRhi backend is **a new `Renderer` subclass**. The only change needed for
containment: add a batched-geometry entry point alongside the existing
immediate-mode one, so retained-mode backends are not forced to fake
immediate mode.

```cpp
// renderer.h — additive, does not break existing immediate-mode renderers
class Renderer {
public:
  // EXISTING immediate-mode path — GLRenderer/POVRenderer keep using it.
  virtual void render_(const GeometricObject*);
  virtual void renderSphere_(const Sphere&);
  // … per-primitive virtuals stay …

  // NEW retained-mode path — a backend overrides EITHER this OR the
  // per-primitive virtuals. Default impl just fans out to render_() so
  // existing backends are untouched.
  virtual void bufferRepresentation(const Representation&) {}   // already exists
  virtual void renderRepresentations_(const RepresentationList&); // new, batched

  // Capability query so RenderSetup/Scene branch on capability, not on type.
  struct Caps { bool retained_mode; bool offscreen; bool picking; bool stereo; };
  virtual Caps capabilities() const = 0;
};
```

Net effect: a QRhi backend implements `renderRepresentations_()` +
`capabilities()` and ignores the per-primitive virtuals. `GLRenderer` is
untouched.

### 2. `RenderSurface` — extract the context/widget concern out of `GLRenderWindow`

`RenderTarget` is already the right shape. Promote it (or a thin subclass
`RenderSurface`) to own the *context lifecycle* verbs that currently leak
through `GLRenderWindow` + `RenderSetup::makeCurrent()`:

```cpp
class RenderSurface : public RenderTarget {
public:
  // Replaces RenderSetup::makeCurrent()'s GL-specific body.
  // GL backend → makeCurrent(); QRhi backend → begin command buffer; no-op for POV/STL.
  virtual void beginFrame() = 0;
  virtual void endFrame()   = 0;          // GL → implicit swap; QRhi → submit
  virtual void* nativeHandle() = 0;       // opaque; only the matching backend casts it
  // getBuffer/getFormat/resize/refresh/prepareRendering already inherited.
};
```

Concrete surfaces, each in its own file, none named by `Scene`:
- `QtOpenGLSurface`  — wraps `QOpenGLWidget` (today's `GLRenderWindow`, post Phase 2)
- `QtRhiSurface`     — wraps `QRhiWidget` (Phase 5, if QRhi is chosen)
- `OffscreenSurface` — FBO / CPU buffer for the raytracer + PNG export
- the existing `GLOffscreenTarget` folds into `OffscreenSurface`

### 3. `RendererFactory` — `Scene` constructs by enum, never by `new GLRenderWindow`

```cpp
namespace RendererFactory {
  enum class Kind { OpenGL_Fixed, OpenGL_Core, QRhi, Raytracer, POV, STL, VRML };
  std::unique_ptr<Renderer>      makeRenderer(Kind);
  std::unique_ptr<RenderSurface> makeSurface(Kind, QWidget* parent);
}
```

`Scene::registerRenderers_()` becomes a loop over `Kind`s — and the Phase 5
swap is **one new `case` in the factory plus two new files**. `scene.C`
loses every `dynamic_cast<GLRenderWindow>` / `dynamic_cast<GLRenderer>` and
every `new GLRenderWindow`.

## What the Phase 5 swap then looks like

With the boundary in place, "4b" (GL-core **or** QRhi) is contained to:

1. Add `OpenGL_Core` (or `QRhi`) to `RendererFactory::Kind`.
2. New file: `coreGLRenderer.C` / `rhiRenderer.C` — a `Renderer` subclass
   implementing `renderRepresentations_()` + `capabilities()`.
3. New file: `qtRhiSurface.C` (only if QRhi) — a `RenderSurface` subclass.
4. **Zero changes to `scene.C`, `RenderSetup`, the message bus, or the
   widget tree.** `Scene` already talks only to `Renderer` / `RenderSurface`.

Without the boundary, Phase 5 re-touches `scene.C` (~3300 lines),
`renderSetup.C`, and the widget hierarchy — i.e. it repeats Phase 2's blast
radius. With it, Phase 5 is ~2 new files behind a `BALL_RENDERER` option.

## Sequencing recommendation

This boundary work is small but **must not** be jammed into Phase 2 — Phase 2
is already the highest-risk phase and is at its human-verify checkpoint. Two
options:

- **(Recommended) New phase 2.5 "Renderer boundary extraction"** — insert
  between Phase 2 and Phase 5 via `/gsd-insert-phase`. Pure refactor, no
  behaviour change, fully `grep`/compile-verifiable. ~3–4 tasks: extract
  `RenderSurface`, add `RendererFactory`, delete the `dynamic_cast`s, move
  `makeCurrent` behind `beginFrame/endFrame`. Landing it makes Phase 5
  estimable and contained.
- **Or fold into the front of Phase 5** — acceptable, but then Phase 5's plan
  must explicitly split "boundary extraction" (wave 1) from "new backend"
  (wave 2+), or it sprawls.

Either way: the boundary extraction is a **prerequisite of**, not a part of,
the actual pipeline rewrite.

## Out of scope for this draft

- Choosing GL-core vs QRhi — that decision belongs in Phase 5 research
  (`ROADMAP-1.6.md` Phase 4b already frames the tradeoff: QRhi removes all
  per-OS graphics code, GL-core is the smaller rewrite).
- The actual shader/pipeline code for the new backend.
- Touching `POVRenderer`/`STLRenderer`/`VRMLRenderer` — they already fit the
  `Renderer` interface and need no boundary work.
