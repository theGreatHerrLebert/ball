// -*- Mode: C++; tab-width: 2; -*-
// vi: set ts=2:
//

#include <BALL/VIEW/RENDERING/rendererFactory.h>

#include <BALL/VIEW/RENDERING/RENDERERS/glRenderer.h>
#include <BALL/VIEW/RENDERING/RENDERERS/POVRenderer.h>
#include <BALL/VIEW/RENDERING/RENDERERS/STLRenderer.h>
#include <BALL/VIEW/RENDERING/RENDERERS/VRMLRenderer.h>
#include <BALL/VIEW/RENDERING/glRenderWindow.h>

#ifdef BALL_HAS_RTFACT
# include <BALL/VIEW/RENDERING/RENDERERS/rtfactRenderer.h>
#endif

namespace BALL
{
	namespace VIEW
	{
		namespace RendererFactory
		{

			Renderer* makeRenderer(Kind kind)
			{
				switch (kind)
				{
					case Kind::OpenGL_Fixed:
						return new GLRenderer;

					case Kind::Raytracer:
#ifdef BALL_HAS_RTFACT
						return new RTfactRenderer();
#else
						// No raytracer backend in this build -- fall back to the
						// fixed-function GL renderer, matching what scene.C's
						// registerRenderers_() #else branch does today.
						return new GLRenderer;
#endif

					case Kind::POV:
						return new POVRenderer;

					case Kind::STL:
						return new STLRenderer;

					case Kind::VRML:
						return new VRMLRenderer;

					default:
						return 0;
				}
			}

			RenderSurface* makeSurface(Kind kind, QWidget* parent)
			{
				switch (kind)
				{
					case Kind::OpenGL_Fixed:
						return new GLRenderWindow(parent);

					case Kind::Raytracer:
						// The raytracer renders into a CPU buffer presented by a
						// GLRenderWindow, exactly as scene.C does today.
						return new GLRenderWindow(parent);

					case Kind::POV:
					case Kind::STL:
					case Kind::VRML:
						// File renderers have no on-screen surface -- scene.C does
						// not construct a window for them. Keep parity.
						return 0;

					default:
						return 0;
				}
			}

		} // namespace RendererFactory

	} // namespace VIEW

} // namespace BALL
