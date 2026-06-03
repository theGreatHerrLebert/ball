// -*- Mode: C++; tab-width: 2; -*-
// vi: set ts=2:
//

#ifndef BALL_VIEW_RENDERING_GLOFFSCREENTARGET_H
#define BALL_VIEW_RENDERING_GLOFFSCREENTARGET_H

#ifndef BALL_COMMON_GLOBAL_H
# include <BALL/COMMON/global.h>
#endif

#ifndef BALL_VIEW_RENDERING_RENDERWINDOW_H
# include <BALL/VIEW/RENDERING/renderWindow.h>
#endif

#ifndef BALL_VIEW_RENDERING_GLRENDERWINDOW_H
# include <BALL/VIEW/RENDERING/glRenderWindow.h>
#endif


#include <QtGui/QPaintDevice>

#include <boost/shared_ptr.hpp>

class QOpenGLFramebufferObject;

namespace BALL
{
	namespace VIEW
	{				
		/** Model of the \link RenderWindow \endlink which uses OpenGL to render its buffer into a file.
		 * 
		 *  The class will try to use a pixel buffer target if this is available. If the OpenGL driver has
		 *  no support for PixelBufferObjects, we try to render into a window on screen instead.
		 */
		class BALL_VIEW_EXPORT GLOffscreenTarget
			: public RenderWindow,
				public QPaintDevice
		{
			public:
				/** Create a new GLOffscreenTarget with context shared from an existing GLRenderWindow.
				 */
				GLOffscreenTarget(GLRenderWindow* share_from, const String& filename);

				virtual void prepareRendering();
				virtual void prepareUpscaling(Size final_width, Size final_height);

				/* RenderSurface methods -- context-lifecycle verbs.
				 * An FBO has no context of its own; beginFrame() makes the
				 * shared GLRenderWindow context current (same as prepareRendering()).
				 */
				virtual void beginFrame() override;
				virtual void endFrame() override;
				virtual void* nativeHandle() override;

				virtual bool resize(const unsigned int width, const unsigned int height);
				virtual void refresh();

				void tryUsePixelBuffer(bool use_pbo = true);

				QImage getImage();
				void updateImageTile(Size x_lower, Size y_lower, Size x_upper, Size y_upper);

				virtual QPaintEngine* paintEngine() const;
				virtual int metric(PaintDeviceMetric metric) const;

			protected:
				String     		 filename_;

				GLRenderWindow* share_from_;

				boost::shared_ptr<QOpenGLFramebufferObject> fbo_;

				QImage current_image_;
		};

	}
}
#endif // BALL_VIEW_RENDERING_GLOFFSCREENTARGET_H
