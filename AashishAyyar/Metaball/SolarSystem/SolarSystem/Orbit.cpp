#include "Common.h"

#define ELLIPSE_SPLICES 1000
#define ORBIT_INCREMENT 1000

#define SCALE_MERCURY_ORBIT (30.0f							)
#define SCALE_VENUS_ORBIT	(SCALE_MERCURY_ORBIT +	20.0f	)
#define SCALE_EARTH_ORBIT	(SCALE_VENUS_ORBIT   +	20.0f	)
#define SCALE_MARS_ORBIT	(SCALE_EARTH_ORBIT   +  20.0f	)
#define SCALE_JUPITER_ORBIT (SCALE_MARS_ORBIT    +  20.0f	)
#define SCALE_SATURN_ORBIT	(SCALE_JUPITER_ORBIT +  25.0f	)
#define SCALE_URANUS_ORBIT	(SCALE_SATURN_ORBIT  +  20.0f	)
#define SCALE_NEPTUNE_ORBIT (SCALE_URANUS_ORBIT  +  20.0f	)
#define SCALE_PLUTO_ORBIT	(SCALE_NEPTUNE_ORBIT +  20.0f	)

extern ORBIT gOrbit;
extern COLOR_SHADER gColorShader;
extern mat4 gPerspectiveProjectionMatrix;
extern mat4 gViewMatrix;

FLOAT gfPlanetAngle = 0.0f;
static FLOAT gIndex = 0.0f;

static BOOL InitVertexArrayAndBuffers(ORBIT &Orbit);

BOOL InitOrbit(ORBIT &Orbit, UINT uiMajorAxis, UINT uiMinorAxis) 
{
	BOOL bRetVal = FALSE;

	bRetVal = GetEllipseData(ELLIPSE_SPLICES, uiMajorAxis, uiMinorAxis, Orbit.EllipseData);
	if (!bRetVal)
	{
		CleanupOrbit(Orbit);
		return FALSE;
	}

	bRetVal = InitVertexArrayAndBuffers(Orbit);
	if (!bRetVal)
	{
		CleanupOrbit(Orbit);
		return FALSE;
	}



	return TRUE;
}

static BOOL InitVertexArrayAndBuffers(ORBIT &Orbit)
{
	glGenVertexArrays(1, &Orbit.vaoOrbit);
	glBindVertexArray(Orbit.vaoOrbit);

	//
	//	Position
	//
	glGenBuffers(1, &Orbit.vboOrbitPosition);
	glBindBuffer(GL_ARRAY_BUFFER, Orbit.vboOrbitPosition);
	glBufferData(GL_ARRAY_BUFFER, Orbit.EllipseData.uiVerticesCount * 3 * sizeof(float), Orbit.EllipseData.pfVerticesEllipse, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//
	//	Color
	//
	glGenBuffers(1, &Orbit.vboOrbitColor);
	glBindBuffer(GL_ARRAY_BUFFER, Orbit.vboOrbitColor);
	glBufferData(GL_ARRAY_BUFFER, Orbit.EllipseData.uiVerticesCount * 3 * sizeof(float), Orbit.EllipseData.pfColorsEllipse, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	return TRUE;
}

void DrawOrbit(ORBIT &Orbit, COLOR_SHADER &ColorShader, vmath::mat4 &mvpMatrix) 
{
	glUseProgram(ColorShader.Shader.renderProgram);
	glUniformMatrix4fv(ColorShader.mvpUniform, 1, GL_FALSE, mvpMatrix);
		glBindVertexArray(Orbit.vaoOrbit);
			glDrawArrays(GL_LINE_STRIP, 0, Orbit.EllipseData.uiVerticesCount);
		glBindVertexArray(0);
	glUseProgram(0);
}

void CleanupOrbit(ORBIT &Orbit) 
{
	if (Orbit.vaoOrbit)
	{
		glDeleteVertexArrays(1, &Orbit.vaoOrbit);
		Orbit.vaoOrbit = 0;
	}

	if (Orbit.vboOrbitPosition)
	{
		glDeleteBuffers(1, &Orbit.vboOrbitPosition);
		Orbit.vboOrbitPosition = 0;
	}

	if (Orbit.vboOrbitColor)
	{
		glDeleteBuffers(1, &Orbit.vboOrbitColor);
		Orbit.vboOrbitColor = 0;
	}

	CleanupEllipseData(Orbit.EllipseData);
}

void UpdatePlanetsAngle() 
{
	gIndex += ROTATE_SPEED;
	if (gIndex > ORBIT_INCREMENT)
		gIndex = 0.0f;
	
	gfPlanetAngle = (2.0f * PI) * ((FLOAT)gIndex / ((FLOAT)ORBIT_INCREMENT));
}

void GetEllipseRadius(PLANETS_AND_SATELLITES PlanetEnum, FLOAT &fMajorAxis, FLOAT &fMinorAxis)
{
	switch (PlanetEnum) 
	{
	case MERCURY:
		fMajorAxis = DEFAULT_ELLIPSE_MAJOR_RADIUS * SCALE_MERCURY_ORBIT;
		fMinorAxis = DEFAULT_ELLIPSE_MINOR_RADIUS * SCALE_MERCURY_ORBIT;
	break;
	case VENUS:
		fMajorAxis = DEFAULT_ELLIPSE_MAJOR_RADIUS * SCALE_VENUS_ORBIT;
		fMinorAxis = DEFAULT_ELLIPSE_MINOR_RADIUS * SCALE_VENUS_ORBIT;
	break;
	case EARTH:
		fMajorAxis = DEFAULT_ELLIPSE_MAJOR_RADIUS * SCALE_EARTH_ORBIT;
		fMinorAxis = DEFAULT_ELLIPSE_MINOR_RADIUS * SCALE_EARTH_ORBIT;
	break;
	case MARS:
		fMajorAxis = DEFAULT_ELLIPSE_MAJOR_RADIUS * SCALE_MARS_ORBIT;
		fMinorAxis = DEFAULT_ELLIPSE_MINOR_RADIUS * SCALE_MARS_ORBIT;
	break;
	case JUPITER:
		fMajorAxis = DEFAULT_ELLIPSE_MAJOR_RADIUS * SCALE_JUPITER_ORBIT;
		fMinorAxis = DEFAULT_ELLIPSE_MINOR_RADIUS * SCALE_JUPITER_ORBIT;
	break;
	case SATURN:
		fMajorAxis = DEFAULT_ELLIPSE_MAJOR_RADIUS * SCALE_SATURN_ORBIT;
		fMinorAxis = DEFAULT_ELLIPSE_MINOR_RADIUS * SCALE_SATURN_ORBIT;
	break;
	case URANUS:
		fMajorAxis = DEFAULT_ELLIPSE_MAJOR_RADIUS * SCALE_URANUS_ORBIT;
		fMinorAxis = DEFAULT_ELLIPSE_MINOR_RADIUS * SCALE_URANUS_ORBIT;
	break;
	case NEPTUNE:
		fMajorAxis = DEFAULT_ELLIPSE_MAJOR_RADIUS * SCALE_NEPTUNE_ORBIT;
		fMinorAxis = DEFAULT_ELLIPSE_MINOR_RADIUS * SCALE_NEPTUNE_ORBIT;
	break;
	case PLUTO:
		fMajorAxis = DEFAULT_ELLIPSE_MAJOR_RADIUS * SCALE_PLUTO_ORBIT;
		fMinorAxis = DEFAULT_ELLIPSE_MINOR_RADIUS * SCALE_PLUTO_ORBIT;
	break;
	}

	return;
}

FLOAT GetPlanetXPosition(ORBIT &Orbit, PLANETS_AND_SATELLITES PlanetEnum, FLOAT fAngle)
{
	FLOAT fMajorAxisRadius = 0.0f;
	FLOAT fMinorAxisRadius = 0.0f;
	
	GetEllipseRadius(PlanetEnum, fMajorAxisRadius, fMinorAxisRadius);

	return fMajorAxisRadius * (FLOAT)cos(fAngle);
}

FLOAT GetPlanetZPosition(ORBIT &Orbit, PLANETS_AND_SATELLITES PlanetEnum, FLOAT fAngle)
{
	FLOAT fMajorAxisRadius = 0.0f;
	FLOAT fMinorAxisRadius = 0.0f;

	GetEllipseRadius(PlanetEnum, fMajorAxisRadius, fMinorAxisRadius);

	return -(fMinorAxisRadius * (FLOAT)sin(fAngle));
}

void DrawAllOrbits()
{
	mat4 scaleMatrix = mat4::identity();
	mat4 modelViewMatrix = mat4::identity();
	mat4 modelViewProjectionMatrix = mat4::identity();

	// Mercury
	scaleMatrix = scale(SCALE_MERCURY_ORBIT, SCALE_MERCURY_ORBIT, SCALE_MERCURY_ORBIT);
	modelViewMatrix = scaleMatrix;
	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * gViewMatrix * modelViewMatrix;
	DrawOrbit(gOrbit, gColorShader, modelViewProjectionMatrix);

	// Venus
	scaleMatrix = scale(SCALE_VENUS_ORBIT, SCALE_VENUS_ORBIT, SCALE_VENUS_ORBIT);
	modelViewMatrix = scaleMatrix;
	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * gViewMatrix * modelViewMatrix;
	DrawOrbit(gOrbit, gColorShader, modelViewProjectionMatrix);

	// Earth
	scaleMatrix = scale(SCALE_EARTH_ORBIT, SCALE_EARTH_ORBIT, SCALE_EARTH_ORBIT);
	modelViewMatrix = scaleMatrix;
	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * gViewMatrix * modelViewMatrix;
	DrawOrbit(gOrbit, gColorShader, modelViewProjectionMatrix);

	// Mars
	scaleMatrix = scale(SCALE_MARS_ORBIT, SCALE_MARS_ORBIT, SCALE_MARS_ORBIT);
	modelViewMatrix = scaleMatrix;
	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * gViewMatrix * modelViewMatrix;
	DrawOrbit(gOrbit, gColorShader, modelViewProjectionMatrix);

	// Jupiter
	scaleMatrix = scale(SCALE_JUPITER_ORBIT, SCALE_JUPITER_ORBIT, SCALE_JUPITER_ORBIT);
	modelViewMatrix = scaleMatrix;
	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * gViewMatrix * modelViewMatrix;
	DrawOrbit(gOrbit, gColorShader, modelViewProjectionMatrix);

	// Saturn
	scaleMatrix = scale(SCALE_SATURN_ORBIT, SCALE_SATURN_ORBIT, SCALE_SATURN_ORBIT);
	modelViewMatrix = scaleMatrix;
	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * gViewMatrix * modelViewMatrix;
	DrawOrbit(gOrbit, gColorShader, modelViewProjectionMatrix);

	// Uranus
	scaleMatrix = scale(SCALE_URANUS_ORBIT, SCALE_URANUS_ORBIT, SCALE_URANUS_ORBIT);
	modelViewMatrix = scaleMatrix;
	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * gViewMatrix * modelViewMatrix;
	DrawOrbit(gOrbit, gColorShader, modelViewProjectionMatrix);

	// Neptune
	scaleMatrix = scale(SCALE_NEPTUNE_ORBIT, SCALE_NEPTUNE_ORBIT, SCALE_NEPTUNE_ORBIT);
	modelViewMatrix = scaleMatrix;
	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * gViewMatrix * modelViewMatrix;
	DrawOrbit(gOrbit, gColorShader, modelViewProjectionMatrix);

	// Pluto
	scaleMatrix = scale(SCALE_PLUTO_ORBIT, SCALE_PLUTO_ORBIT, SCALE_PLUTO_ORBIT);
	modelViewMatrix = scaleMatrix;
	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * gViewMatrix * modelViewMatrix;
	DrawOrbit(gOrbit, gColorShader, modelViewProjectionMatrix);
}