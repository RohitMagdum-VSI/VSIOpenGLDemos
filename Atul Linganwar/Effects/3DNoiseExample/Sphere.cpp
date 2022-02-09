#include <Windows.h>
#include <stdio.h>
#include <gl/glew.h>
#include <gl/GL.h>

#include "../common/vmath.h"
#include "../common/Common.h"
#include "../common/Shapes.h"

#include "Sphere.h"

bool InitializeSphere(GLfloat fRadius, GLint iSlices, GLint iStacks, SPHERE* pSphere)
{
	BOOL boRet = FALSE;

	boRet = GetSphereData(fRadius, iSlices, iStacks, pSphere->SphereData, FALSE);
	if (boRet == FALSE)
	{
		return false;
	}

	glGenVertexArrays(1, &pSphere->uiVAO);
	glBindVertexArray(pSphere->uiVAO);

	// Position
	glGenBuffers(1, &pSphere->uiVBOPosition);
	glBindBuffer(GL_ARRAY_BUFFER, pSphere->uiVBOPosition);
	glBufferData(
		GL_ARRAY_BUFFER,
		pSphere->SphereData.uiVerticesCount * 3 * sizeof(float),
		pSphere->SphereData.pfVerticesSphere,
		GL_STATIC_DRAW
	);
	glVertexAttribPointer(OGL_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Normals
	glGenBuffers(1, &pSphere->uiVBONormal);
	glBindBuffer(GL_ARRAY_BUFFER, pSphere->uiVBONormal);
	glBufferData(
		GL_ARRAY_BUFFER,
		pSphere->SphereData.uiVerticesCount * 3 * sizeof(float),
		pSphere->SphereData.pfNormalsSphere,
		GL_STATIC_DRAW
	);
	glVertexAttribPointer(OGL_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_NORMAL);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Texture
	glGenBuffers(1, &pSphere->uiVBOTexture);
	glBindBuffer(GL_ARRAY_BUFFER, pSphere->uiVBOTexture);
	glBufferData(
		GL_ARRAY_BUFFER,
		pSphere->SphereData.uiVerticesCount * 2 * sizeof(float),
		pSphere->SphereData.pfTexCoordsSphere,
		GL_STATIC_DRAW
	);
	glVertexAttribPointer(OGL_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_TEXTURE0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Elements
	glGenBuffers(1, &pSphere->uiVBOElements);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pSphere->uiVBOElements);
	glBufferData(
		GL_ELEMENT_ARRAY_BUFFER,
		pSphere->SphereData.uiIndicesCount * sizeof(UINT),
		pSphere->SphereData.puiIndicesSphere,
		GL_STATIC_DRAW
	);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	// Free sphere data
	CleanupSphereData(pSphere->SphereData);
}

void DrawSphere(SPHERE* pSphere)
{
	glBindVertexArray(pSphere->uiVAO);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pSphere->uiVBOElements);
	glDrawElements(GL_TRIANGLES, pSphere->SphereData.uiIndicesCount, GL_UNSIGNED_INT, 0);

	glBindVertexArray(0);
}

void FreeSphere(SPHERE* pSphere)
{
	// Note: do not free sphere data. (already freed after processing vao).

	if (pSphere->uiVAO)
	{
		glDeleteVertexArrays(1, &pSphere->uiVAO);
		pSphere->uiVAO = 0;
	}

	if (pSphere->uiVBOPosition)
	{
		glDeleteBuffers(1, &pSphere->uiVBOPosition);
		pSphere->uiVBOPosition = 0;
	}

	if (pSphere->uiVBOTexture)
	{
		glDeleteBuffers(1, &pSphere->uiVBOTexture);
		pSphere->uiVBOTexture = 0;
	}

	if (pSphere->uiVBONormal)
	{
		glDeleteBuffers(1, &pSphere->uiVBONormal);
		pSphere->uiVBONormal = 0;
	}

	if (pSphere->uiVBOElements)
	{
		glDeleteBuffers(1, &pSphere->uiVBOElements);
		pSphere->uiVBOElements = 0;
	}

	return;
}
