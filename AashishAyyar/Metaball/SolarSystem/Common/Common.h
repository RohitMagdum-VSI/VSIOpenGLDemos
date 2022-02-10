#pragma once

#include <windows.h>
#include <stdio.h>

#include <gl/glew.h>
#include <gl/GL.h>

#include "vmath.h"

//
//	MACROS
//
#define ROTATE_SPEED 0.1f

#define ID_STARRY_SKY		101
#define ID_PLANET_SUN		102
#define ID_PLANET_MECURY	103
#define ID_PLANET_VENUS		104
#define ID_PLANET_EARTH		105
#define ID_PLANET_MARS		106
#define ID_PLANET_JUPITER	107
#define ID_PLANET_SATURN	108
#define ID_PLANET_URANUS	109
#define ID_PLANET_NEPTUNE	110
#define ID_PLANET_PLUTO		111
#define ID_SATURN_RING		112

#define PI 3.14159265358979323846264338327950288f

#define DEFAULT_PLANET_RADIUS 100.0f
#define DEFAULT_ELLIPSE_MAJOR_RADIUS 120.0f
#define DEFAULT_ELLIPSE_MINOR_RADIUS 60.0f

#define ANGLE_WITH_OFFSET(ANGLE, OFFSET) (FLOAT)((((ANGLE + OFFSET) > 360.0f) ? ((ANGLE + OFFSET) - 360.0f) : (ANGLE + OFFSET)))

#define MERCURY_OFFSET	150.0f
#define VENUS_OFFSET	20.0f	 
#define EARTH_OFFSET	75.0f	 
#define MARS_OFFSET		200.0f
#define JUPITER_OFFSET  270.0f
#define SATURN_OFFSET	190.0f 
#define URANUS_OFFSET	300.0f
#define NEPTUNE_OFFSET  120.0f
#define PLUTO_OFFSET	340.0f 

//
//	 ENUMERATIONS
//
enum PLANETS_AND_SATELLITES
{
	NONE = 0,
	SUN,
	MERCURY,
	VENUS,
	EARTH,
	MARS,
	JUPITER,
	SATURN,
	URANUS,
	NEPTUNE,
	PLUTO,
	MOON,
	SPHERE_MAP,
	SATURN_RING_ID
};

enum
{
	OGL_ATTRIBUTE_POSITION = 0,
	OGL_ATTRIBUTE_COLOR,
	OGL_ATTRIBUTE_NORMAL,
	OGL_ATTRIBUTE_TEXTURE0,
};

#include "FrameBuffer.h"
#include "Shapes.h"
#include "Sphere.h"
#include "Ring.h"
#include "Shader.h"
#include "Planet.h"
#include "Orbit.h"
#include "KtxLoader.h"
#include "StarField.h"
#include "Picking.h"
#include "Saturn.h"
#include "PlanetDetails.h"

//
//	NAMESPACES
//
using namespace vmath;

//
// FUCNTION PROTOTYPE
//
int LoadGLTextures(GLuint *texture, TCHAR imageResourceId[]);
