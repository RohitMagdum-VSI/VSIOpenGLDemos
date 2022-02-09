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

	boRet = GetSphereData(fRadius, iSlices, iStacks, pPlanet->Sphere.SphereData, FALSE);
	if (boRet == FALSE)
	{
		return false;
	}

	glGenVertexArrays(1, &pPlanet->Sphere.uiVAO);
	glBindVertexArray(pPlanet->Sphere.uiVAO);

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

	// Normals
	glGenBuffers(1, &pPlanet->Sphere.uiVBONormal);
	glBindBuffer(GL_ARRAY_BUFFER, pPlanet->Sphere.uiVBONormal);
	glBufferData(
		GL_ARRAY_BUFFER,
		pPlanet->Sphere.SphereData.uiVerticesCount * 3 * sizeof(float),
		pPlanet->Sphere.SphereData.pfNormalsSphere,
		GL_STATIC_DRAW
	);
	glVertexAttribPointer(OGL_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_NORMAL);
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

	//
	// VAO for picking
	//
	glGenVertexArrays(1, &pPlanet->uiVAOPicking);
	glBindVertexArray(pPlanet->uiVAOPicking);

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

	glGenBuffers(1, &pPlanet->uiVBOElementsPicking);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pPlanet->uiVBOElementsPicking);
	glBufferData(
		GL_ELEMENT_ARRAY_BUFFER,
		pPlanet->Sphere.SphereData.uiIndicesCount * sizeof(UINT),
		pPlanet->Sphere.SphereData.puiIndicesSphere,
		GL_STATIC_DRAW
	);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	// Free sphere data
	CleanupSphereData(pPlanet->Sphere.SphereData);

	return TRUE;
}

void FreePlanetsData(PLANETS* pPlanet)
{
	// Note: do not free sphere data. (already freed after processing vao).
	
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

void DrawPlanetPicking(PLANETS* pPlanet)
{
	glBindVertexArray(pPlanet->uiVAOPicking);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pPlanet->uiVBOElementsPicking);
	glDrawElements(GL_TRIANGLES, pPlanet->Sphere.SphereData.uiIndicesCount, GL_UNSIGNED_INT, 0);

	glBindVertexArray(0);
}

bool InitializeSaturnRingData(GLfloat fInnerRadius, GLfloat fOuterRadius, SATURN_RING* pRing)
{
	BOOL boRet = FALSE;

	boRet = GetRingData(fInnerRadius, fOuterRadius, 60, pRing->Ring.RingData);
	if (boRet == FALSE)
	{
		return false;
	}

	glGenVertexArrays(1, &pRing->Ring.uiVAO);
	glBindVertexArray(pRing->Ring.uiVAO);

	// Position
	glGenBuffers(1, &pRing->Ring.uiVBOPosition);
	glBindBuffer(GL_ARRAY_BUFFER, pRing->Ring.uiVBOPosition);
	glBufferData(
		GL_ARRAY_BUFFER,
		pRing->Ring.RingData.uiVerticesCount * 3 * sizeof(float),
		pRing->Ring.RingData.pfVerticesRing,
		GL_STATIC_DRAW
	);
	glVertexAttribPointer(OGL_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Normals
	glGenBuffers(1, &pRing->Ring.uiVBONormal);
	glBindBuffer(GL_ARRAY_BUFFER, pRing->Ring.uiVBONormal);
	glBufferData(
		GL_ARRAY_BUFFER,
		pRing->Ring.RingData.uiVerticesCount * 3 * sizeof(float),
		pRing->Ring.RingData.pfNormalsRing,
		GL_STATIC_DRAW
	);
	glVertexAttribPointer(OGL_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_NORMAL);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Texture
	glGenBuffers(1, &pRing->Ring.uiVBOTexture);
	glBindBuffer(GL_ARRAY_BUFFER, pRing->Ring.uiVBOTexture);
	glBufferData(
		GL_ARRAY_BUFFER,
		pRing->Ring.RingData.uiVerticesCount * 2 * sizeof(float),
		pRing->Ring.RingData.pfTexCoordsRing,
		GL_STATIC_DRAW
	);
	glVertexAttribPointer(OGL_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_TEXTURE0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Elements
	glGenBuffers(1, &pRing->Ring.uiVBOElements);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pRing->Ring.uiVBOElements);
	glBufferData(
		GL_ELEMENT_ARRAY_BUFFER,
		pRing->Ring.RingData.uiIndicesCount * sizeof(UINT),
		pRing->Ring.RingData.puiIndicesRing,
		GL_STATIC_DRAW
	);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	//
	// VAO for picking
	//
	glGenVertexArrays(1, &pRing->uiVAOPicking);
	glBindVertexArray(pRing->uiVAOPicking);

	glGenBuffers(1, &pRing->Ring.uiVBOPosition);
	glBindBuffer(GL_ARRAY_BUFFER, pRing->Ring.uiVBOPosition);
	glBufferData(
		GL_ARRAY_BUFFER,
		pRing->Ring.RingData.uiVerticesCount * 3 * sizeof(float),
		pRing->Ring.RingData.pfVerticesRing,
		GL_STATIC_DRAW
	);
	glVertexAttribPointer(OGL_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &pRing->uiVBOElementsPicking);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pRing->uiVBOElementsPicking);
	glBufferData(
		GL_ELEMENT_ARRAY_BUFFER,
		pRing->Ring.RingData.uiIndicesCount * sizeof(UINT),
		pRing->Ring.RingData.puiIndicesRing,
		GL_STATIC_DRAW
	);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	CleanupRingData(pRing->Ring.RingData);

	return TRUE;
}

void FreeSaturnRingData(SATURN_RING* pRing)
{
	if (pRing->Ring.uiVAO)
	{
		glDeleteVertexArrays(1, &pRing->Ring.uiVAO);
		pRing->Ring.uiVAO = 0;
	}

	if (pRing->Ring.uiVBOPosition)
	{
		glDeleteBuffers(1, &pRing->Ring.uiVBOPosition);
		pRing->Ring.uiVBOPosition = 0;
	}

	if (pRing->Ring.uiVBOTexture)
	{
		glDeleteBuffers(1, &pRing->Ring.uiVBOTexture);
		pRing->Ring.uiVBOTexture = 0;
	}

	if (pRing->Ring.uiVBONormal)
	{
		glDeleteBuffers(1, &pRing->Ring.uiVBONormal);
		pRing->Ring.uiVBONormal = 0;
	}

	if (pRing->Ring.uiVBOElements)
	{
		glDeleteBuffers(1, &pRing->Ring.uiVBOElements);
		pRing->Ring.uiVBOElements = 0;
	}
}

void DrawSaturnRing(SATURN_RING* pRing)
{
	glBindVertexArray(pRing->Ring.uiVAO);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pRing->Ring.uiVBOElements);
	glDrawElements(GL_TRIANGLES, pRing->Ring.RingData.uiIndicesCount, GL_UNSIGNED_INT, 0);

	glBindVertexArray(0);
}

void DrawSaturnRingPicking(SATURN_RING* pRing)
{
	glBindVertexArray(pRing->uiVAOPicking);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pRing->uiVBOElementsPicking);
	glDrawElements(GL_TRIANGLES, pRing->Ring.RingData.uiIndicesCount, GL_UNSIGNED_INT, 0);

	glBindVertexArray(0);
}