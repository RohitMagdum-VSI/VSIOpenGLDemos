#include "Common.h"

#define SPHERE_SLICES 300
#define SPHERE_STACKS 300

BOOL InitSphere(FLOAT fRadius, SPHERE &Sphere, BOOL InvertedNormals) 
{
	BOOL bRetVal = FALSE;

	bRetVal = GetSphereData(fRadius, SPHERE_SLICES, SPHERE_STACKS, Sphere.SphereData, InvertedNormals);
	if (bRetVal == FALSE)
	{
		CleanupSphere(Sphere);
		return FALSE;
	}

	bRetVal = InitVertexArrayAndBuffers(Sphere);
	if (bRetVal == FALSE)
	{
		CleanupSphere(Sphere);
		return FALSE;
	}

	//  Freeing unwanted memory
	CleanupSphereData(Sphere.SphereData);

	return TRUE;
}

BOOL InitVertexArrayAndBuffers(SPHERE &Sphere)
{
	glGenVertexArrays(1, &Sphere.vaoSphere);
	glBindVertexArray(Sphere.vaoSphere);

	//
	//	Position
	//
	glGenBuffers(1, &Sphere.vboSpherePosition);
	glBindBuffer(GL_ARRAY_BUFFER, Sphere.vboSpherePosition);
	glBufferData(GL_ARRAY_BUFFER, Sphere.SphereData.uiVerticesCount * 3 * sizeof(float), Sphere.SphereData.pfVerticesSphere, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//
	//	Normal
	//
	glGenBuffers(1, &Sphere.vboSphereNormal);
	glBindBuffer(GL_ARRAY_BUFFER, Sphere.vboSphereNormal);
	glBufferData(GL_ARRAY_BUFFER, Sphere.SphereData.uiVerticesCount * 3 * sizeof(float), Sphere.SphereData.pfNormalsSphere, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_NORMAL);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//
	//	Texture
	//
	glGenBuffers(1, &Sphere.vboSphereTexture);
	glBindBuffer(GL_ARRAY_BUFFER, Sphere.vboSphereTexture);
	glBufferData(GL_ARRAY_BUFFER, Sphere.SphereData.uiVerticesCount * 2 * sizeof(float), Sphere.SphereData.pfTexCoordsSphere, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_TEXTURE0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//
	//	Elements
	//	
	glGenBuffers(1, &Sphere.vboSphereElements);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Sphere.vboSphereElements);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, Sphere.SphereData.uiIndicesCount * sizeof(UINT), Sphere.SphereData.puiIndicesSphere, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	return TRUE;
}

BOOL DrawSphere(SPHERE &Sphere)
{
	// Draw Call
	glBindVertexArray(Sphere.vaoSphere);		
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Sphere.vboSphereElements);
		glDrawElements(GL_TRIANGLES, Sphere.SphereData.uiIndicesCount, GL_UNSIGNED_INT, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	return TRUE;
}

void CleanupSphere(SPHERE &Sphere) 
{
	if (Sphere.vaoSphere)
	{
		glDeleteVertexArrays(1, &Sphere.vaoSphere);
		Sphere.vaoSphere = 0;
	}

	if (Sphere.vboSpherePosition)
	{
		glDeleteBuffers(1, &Sphere.vboSpherePosition);
		Sphere.vboSpherePosition = 0;
	}

	if (Sphere.vboSphereTexture)
	{
		glDeleteBuffers(1, &Sphere.vboSphereTexture);
		Sphere.vboSphereTexture = 0;
	}

	if (Sphere.vboSphereElements)
	{
		glDeleteBuffers(1, &Sphere.vboSphereElements);
		Sphere.vboSphereElements = 0;
	}

	//  Freeing unwanted memory
	CleanupSphereData(Sphere.SphereData);
}
