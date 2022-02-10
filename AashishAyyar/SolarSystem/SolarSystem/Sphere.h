#pragma once

typedef struct _SPHERE 
{
	SPHERE_DATA SphereData;

	GLuint vaoSphere;
	GLuint vboSpherePosition;
	GLuint vboSphereNormal;
	GLuint vboSphereTexture;
	GLuint vboSphereElements;

}SPHERE, *P_SPHERE;

//
//	InitSphere: Allocate and initialize sphere, with specific radius
//
BOOL InitSphere(FLOAT fRadius, SPHERE &Sphere, BOOL InvertedNormals);

BOOL InitVertexArrayAndBuffers(SPHERE &Sphere);

BOOL DrawSphere(SPHERE &Sphere);

void CleanupSphere(SPHERE &Sphere);

