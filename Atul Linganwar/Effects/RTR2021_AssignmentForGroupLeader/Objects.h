#pragma once

typedef struct _SPHERE
{
	SPHERE_DATA SphereData;

	GLuint uiVAO;
	GLuint uiVBOPosition;
	GLuint uiVBOTexture;
	GLuint uiVBONormal;
	GLuint uiVBOElements;

}SPHERE;

typedef struct _RING
{
	RING_DATA RingData;

	GLuint uiVAO;
	GLuint uiVBOPosition;
	GLuint uiVBOTexture;
	GLuint uiVBONormal;
	GLuint uiVBOElements;

}RING;

typedef struct _PLANETS
{
	SPHERE Sphere;

	GLuint uiTexture;

	//vmath::mat4 mat4ModelMatrix;
	//vmath::mat4 mat4ViewMatrix;
	//vmath::mat4 mat4ProjectionMatrix;

	GLuint uiVAOPicking;
	GLuint uiVBOPositionPicking;
	GLuint uiVBOElementsPicking;

	OBJECT_TRANSFORMATION ObjectTransformation;

}PLANETS;

typedef struct _SATURN_RING
{
	RING Ring;

	GLuint uiTexture;

	GLuint uiVAOPicking;
	GLuint uiVBOPositionPicking;
	GLuint uiVBOElementsPicking;

	OBJECT_TRANSFORMATION ObjectTransformation;

}SATURN_RING;

bool InitializePlanetsData(GLfloat fRadius, GLint iSlices, GLint iStacks, PLANETS* pPlanet);
void FreePlanetsData(PLANETS* pPlanet);

void DrawPlanet(PLANETS* pPlanet);
void DrawPlanetPicking(PLANETS* pPlanet);

bool InitializeSaturnRingData(GLfloat fInnerRadius, GLfloat fOuterRadius, SATURN_RING* pRing);
void FreeSaturnRingData(SATURN_RING* pRing);

void DrawSaturnRing(SATURN_RING* pRing);
void DrawSaturnRingPicking(SATURN_RING* pRing);
