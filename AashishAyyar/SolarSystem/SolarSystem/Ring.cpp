#include "Common.h"

#define RING_SLICES 30.0f

BOOL InitRing(FLOAT fInnerRadius, FLOAT fOuterRadius, RING &Ring, BOOL InvertedNormals)
{
	BOOL bRetVal = FALSE;

	bRetVal = GetRingData(fInnerRadius, fOuterRadius, RING_SLICES, Ring.RingData);
	if (bRetVal == FALSE)
	{
		CleanupRing(Ring);
		return FALSE;
	}

	bRetVal = InitVertexArrayAndBuffers(Ring);
	if (bRetVal == FALSE)
	{
		CleanupRing(Ring);
		return FALSE;
	}

	//  Freeing unwanted memory
	CleanupRingData(Ring.RingData);

	return TRUE;
}

BOOL InitVertexArrayAndBuffers(RING &Ring)
{
	glGenVertexArrays(1, &Ring.vaoRing);
	glBindVertexArray(Ring.vaoRing);

	//
	//	Position
	//
	glGenBuffers(1, &Ring.vboRingPosition);
	glBindBuffer(GL_ARRAY_BUFFER, Ring.vboRingPosition);
	glBufferData(GL_ARRAY_BUFFER, Ring.RingData.uiVerticesCount * 3 * sizeof(float), Ring.RingData.pfVerticesRing, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//
	//	Normal
	//
	glGenBuffers(1, &Ring.vboRingNormal);
	glBindBuffer(GL_ARRAY_BUFFER, Ring.vboRingNormal);
	glBufferData(GL_ARRAY_BUFFER, Ring.RingData.uiVerticesCount * 3 * sizeof(float), Ring.RingData.pfNormalsRing, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_NORMAL);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//
	//	Texture
	//
	glGenBuffers(1, &Ring.vboRingTexture);
	glBindBuffer(GL_ARRAY_BUFFER, Ring.vboRingTexture);
	glBufferData(GL_ARRAY_BUFFER, Ring.RingData.uiVerticesCount * 2 * sizeof(float), Ring.RingData.pfTexCoordsRing, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_TEXTURE0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//
	//	Elements
	//	
	glGenBuffers(1, &Ring.vboRingElements);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Ring.vboRingElements);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, Ring.RingData.uiIndicesCount * sizeof(UINT), Ring.RingData.puiIndicesRing, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	return TRUE;
}

BOOL DrawRing(RING &Ring)
{
	// Draw Call
	glBindVertexArray(Ring.vaoRing);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Ring.vboRingElements);
			glDrawElements(GL_TRIANGLES, Ring.RingData.uiIndicesCount, GL_UNSIGNED_INT, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	return TRUE;
}

void CleanupRing(RING &Ring)
{
	if (Ring.vaoRing)
	{
		glDeleteVertexArrays(1, &Ring.vaoRing);
		Ring.vaoRing = 0;
	}

	if (Ring.vboRingPosition)
	{
		glDeleteBuffers(1, &Ring.vboRingPosition);
		Ring.vboRingPosition = 0;
	}

	if (Ring.vboRingTexture)
	{
		glDeleteBuffers(1, &Ring.vboRingTexture);
		Ring.vboRingTexture = 0;
	}

	if (Ring.vboRingElements)
	{
		glDeleteBuffers(1, &Ring.vboRingElements);
		Ring.vboRingElements = 0;
	}

	//  Freeing unwanted memory
	CleanupRingData(Ring.RingData);
}
