// -*- Mode: C++; tab-width: 2; -*-
// vi: set ts=2:
//
// $Id: objectCreator.C,v 1.1.4.1 2007/03/25 22:00:06 oliver Exp $

#include <BALL/CONCEPT/objectCreator.h>

using namespace std;

namespace BALL
{
  
	ObjectCreator::ObjectCreator()
			
		:
		init_(false),
		pm_()
	{
	}

	ObjectCreator::~ObjectCreator()
			
	{
		#ifdef BALL_DEBUG
			cout << "Destructing object " << (void *)this 
				<< " of class " << RTTI::getName<ObjectCreator>() << endl;
		#endif 
	}

	void ObjectCreator::clear()
			
	{
	}


	void ObjectCreator::initPersistenceManager(TextPersistenceManager & /* pm */)
			
	{
	}

	Composite *ObjectCreator::convertObject(PersistentObject & /* po */)
			
	{
		return (Composite *)0;
	}

} // namespace BALL
