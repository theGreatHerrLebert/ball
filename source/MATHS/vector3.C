// -*- Mode: C++; tab-width: 2; -*-
// vi: set ts=2:
//
// $Id: vector3.C,v 1.2 2002/02/27 12:21:29 sturm Exp $

#include <BALL/MATHS/vector3.h>

namespace BALL
{
// Explicit instantiation DEFINITION carries BALL_EXPORT (dllexport on MSVC);
// the matching `extern template` declarations in vector3.h must NOT — that
// combination triggers MSVC C4910.
template class BALL_EXPORT TVector3<float>;

#ifdef BALL_COMPILER_MSVC
	template class BALL_EXPORT std::vector<Vector3>;
#endif

}
