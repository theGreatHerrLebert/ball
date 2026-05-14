// -*- Mode: C++; tab-width: 2; -*-
// vi: set ts=2:
//

#ifndef BALL_VIEW_RENDERING_RENDERSURFACE_H
#define BALL_VIEW_RENDERING_RENDERSURFACE_H

#include <BALL/COMMON/global.h>
#include <BALL/VIEW/RENDERING/renderTarget.h>

namespace BALL
{
	namespace VIEW
	{

		/** Abstract render surface interface.
		 *
		 *  A RenderSurface is a RenderTarget that additionally owns the
		 *  context-lifecycle verbs that used to leak through GLRenderWindow
		 *  and RenderSetup::makeCurrent(). It is intentionally free of any
		 *  Qt-GL types: a GL backend, a QRhi backend, and a file/offscreen
		 *  renderer all model "make ready / finish a frame" differently, but
		 *  callers (RenderSetup, Scene) only ever see the neutral verbs below.
		 */
		class BALL_VIEW_EXPORT RenderSurface : public RenderTarget
		{
			public:

				/// Make this surface ready to receive a frame's draw calls.
				/// GL backend -> makeCurrent(); QRhi backend -> begin command buffer; no-op for file renderers.
				virtual void beginFrame() = 0;

				/// Finish the frame. GL -> implicit swap; QRhi -> submit.
				virtual void endFrame() = 0;

				/// Opaque native handle; only the matching backend casts it.
				virtual void* nativeHandle() = 0;
		};

	} //namespace VIEW

} // namespace BALL

#endif // BALL_VIEW_RENDERING_RENDERSURFACE_H
