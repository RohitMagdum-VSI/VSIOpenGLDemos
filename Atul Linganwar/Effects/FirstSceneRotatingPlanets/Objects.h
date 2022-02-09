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

typedef struct _PLANETS
{
	SPHERE Sphere;

	GLuint uiTexture;

	//vmath::mat4 mat4ModelMatrix;
	//vmath::mat4 mat4ViewMatrix;
	//vmath::mat4 mat4ProjectionMatrix;

	OBJECT_TRANSFORMATION ObjectTransformation;

}PLANETS;

bool InitializePlanetsData(GLfloat fRadius, GLint iSlices, GLint iStacks, PLANETS* pPlanet);
void FreePlanetsData(PLANETS* pPlanet);

void DrawPlanet(PLANETS* pPlanet);
