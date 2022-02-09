#include <Windows.h>
#include <stdio.h>
#include <gl/glew.h>
#include <gl/GL.h>

#include "../common/vmath.h"
#include "../common/Common.h"
#include "../common/Shapes.h"
#include "ObjectTransformations.h"
#include "Objects.h"

bool InitializePlanetsData(GLfloat fRadius, GLint iSlices, GLint iStacks, PLANETS* pPlanet)
{
	BOOL boRet = FALSE;

	boRet = GetSphereData(fRadius, iSlices, iStacks, pPlanet->Sphere.SphereData);
	if (boRet == FALSE)
	{
		return false;
	}

	// Position
	glGenBuffers(1, &pPlanet->Sphere.uiVBOPosition);
	glBindBuffer(GL_ARRAY_BUFFER, pPlanet->Sphere.uiVBOPosition);
	glBufferData(
		GL_ARRAY_BUFFER, 
		pPlanet->Sphere.SphereData.uiVerticesCount * 3 * sizeof(float), 
		pPlanet->Sphere.SphereData.pfVerticesSphere, 
		GL_STATIC_DRAW
	);
	glVertexAttribPointer(OGL_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Texture
	glGenBuffers(1, &pPlanet->Sphere.uiVBOTexture);
	glBindBuffer(GL_ARRAY_BUFFER, pPlanet->Sphere.uiVBOTexture);
	glBufferData(
		GL_ARRAY_BUFFER, 
		pPlanet->Sphere.SphereData.uiVerticesCount * 2 * sizeof(float), 
		pPlanet->Sphere.SphereData.pfTexCoordsSphere, 
		GL_STATIC_DRAW
	);
	glVertexAttribPointer(OGL_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_TEXTURE0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Elements
	glGenBuffers(1, &pPlanet->Sphere.uiVBOElements);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pPlanet->Sphere.uiVBOElements);
	glBufferData(
		GL_ELEMENT_ARRAY_BUFFER, 
		pPlanet->Sphere.SphereData.uiIndicesCount * sizeof(UINT), 
		pPlanet->Sphere.SphereData.puiIndicesSphere, 
		GL_STATIC_DRAW
	);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	//pPlanet->mat4ModelMatrix = vmath::mat4::identity();
	//pPlanet->mat4ViewMatrix = vmath::mat4::identity();
	//pPlanet->mat4ProjectionMatrix = vmath::mat4::identity();

	// Free sphere data
	CleanupSphereData(pPlanet->Sphere.SphereData);

	return TRUE;
}

void FreePlanetsData(PLANETS* pPlanet)
{
	// Note: do not free sphere data. (already freed after processing vao.
	
	if (pPlanet->Sphere.uiVAO)
	{
		glDeleteVertexArrays(1, &pPlanet->Sphere.uiVAO);
		pPlanet->Sphere.uiVAO = 0;
	}

	if (pPlanet->Sphere.uiVBOPosition)
	{
		glDeleteBuffers(1, &pPlanet->Sphere.uiVBOPosition);
		pPlanet->Sphere.uiVBOPosition = 0;
	}

	if (pPlanet->Sphere.uiVBOTexture)
	{
		glDeleteBuffers(1, &pPlanet->Sphere.uiVBOTexture);
		pPlanet->Sphere.uiVBOTexture = 0;
	}

	if (pPlanet->Sphere.uiVBONormal)
	{
		glDeleteBuffers(1, &pPlanet->Sphere.uiVBONormal);
		pPlanet->Sphere.uiVBONormal = 0;
	}

	if (pPlanet->Sphere.uiVBOElements)
	{
		glDeleteBuffers(1, &pPlanet->Sphere.uiVBOElements);
		pPlanet->Sphere.uiVBOElements = 0;
	}

	return;
}

void DrawPlanet(PLANETS* pPlanet)
{
	glBindVertexArray(pPlanet->Sphere.uiVAO);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pPlanet->Sphere.uiVBOElements);
	glDrawElements(GL_TRIANGLES, pPlanet->Sphere.SphereData.uiIndicesCount, GL_UNSIGNED_INT, 0);

	glBindVertexArray(0);
}