#include <Windows.h>
#include <stdio.h>
#include <gl/glew.h>
#include <gl/GL.h>

#include "../common/vmath.h"
#include "../common/Common.h"
#include "../common/Shapes.h"
#include "ObjectTransformations.h"
#include "BasicTextureShader.h"
#include "SphereMap.h"

extern vmath::mat4 gPerspectiveProjectionMatrix;

bool InitializeSphereMapData(GLfloat fRadius, GLint iSlices, GLint iStacks, SPHERE_MAP* pSphereMap)
{
	BOOL boRet = FALSE;

	boRet = GetSphereData(fRadius, iSlices, iStacks, pSphereMap->SphereData, TRUE);
	if (boRet == FALSE)
	{
		return false;
	}

	glGenVertexArrays(1, &pSphereMap->uiVAO);
	glBindVertexArray(pSphereMap->uiVAO);

	// Position
	glGenBuffers(1, &pSphereMap->uiVBOPosition);
	glBindBuffer(GL_ARRAY_BUFFER, pSphereMap->uiVBOPosition);
	glBufferData(
		GL_ARRAY_BUFFER,
		pSphereMap->SphereData.uiVerticesCount * 3 * sizeof(float),
		pSphereMap->SphereData.pfVerticesSphere,
		GL_STATIC_DRAW
	);
	glVertexAttribPointer(OGL_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Texture
	glGenBuffers(1, &pSphereMap->uiVBOTexture);
	glBindBuffer(GL_ARRAY_BUFFER, pSphereMap->uiVBOTexture);
	glBufferData(
		GL_ARRAY_BUFFER,
		pSphereMap->SphereData.uiVerticesCount * 2 * sizeof(float),
		pSphereMap->SphereData.pfTexCoordsSphere,
		GL_STATIC_DRAW
	);
	glVertexAttribPointer(OGL_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_TEXTURE0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Elements
	glGenBuffers(1, &pSphereMap->uiVBOElements);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pSphereMap->uiVBOElements);
	glBufferData(
		GL_ELEMENT_ARRAY_BUFFER,
		pSphereMap->SphereData.uiIndicesCount * sizeof(UINT),
		pSphereMap->SphereData.puiIndicesSphere,
		GL_STATIC_DRAW
	);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	// Free sphere data
	CleanupSphereData(pSphereMap->SphereData);

	return TRUE;
}

void DrawSphereMap(BASIC_TEXTURE_SHADER ShaderProgObj, SPHERE_MAP* pSphereMap, GLuint uiTexture)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glDisable(GL_CULL_FACE);
	glUseProgram(ShaderProgObj.ShaderObject.uiShaderProgramObject);

	vmath::mat4 ModelMatrix = vmath::mat4::identity();
	vmath::mat4 ViewMatrix = vmath::mat4::identity();
	vmath::mat4 TranslationMatrix = vmath::mat4::identity();
	vmath::mat4 RotationMatrix = vmath::mat4::identity();
	vmath::mat4 ScaleMatrix = vmath::mat4::identity();

	ViewMatrix = vmath::lookat(vmath::vec3(0.0f, 0.0f, 9000.0f), vmath::vec3(0.0f, 0.0f, 0.0f), vmath::vec3(0.0f, 1.0f, 0.0f));

	RotationMatrix = vmath::rotate(90.0f, 0.0f, 0.0f, 1.0f);
	RotationMatrix = RotationMatrix * vmath::rotate(270.0f, 0.0f, 1.0f, 0.0f);
	ModelMatrix = ModelMatrix * RotationMatrix;

	ScaleMatrix = vmath::scale(SPHERE_MAP_SCALE);
	ModelMatrix = ModelMatrix * ScaleMatrix;

	glUniformMatrix4fv(ShaderProgObj.uiModelMatrixUniform, 1, GL_FALSE, ModelMatrix);
	glUniformMatrix4fv(ShaderProgObj.uiViewMatrixUniform, 1, GL_FALSE, ViewMatrix);
	glUniformMatrix4fv(ShaderProgObj.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, uiTexture);
	glUniform1i(ShaderProgObj.uiTextureSamplerUniform, 0);

	glBindVertexArray(pSphereMap->uiVAO);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pSphereMap->uiVBOElements);
	glDrawElements(GL_TRIANGLES, pSphereMap->SphereData.uiIndicesCount, GL_UNSIGNED_INT, 0);

	glBindVertexArray(0);

	glUseProgram(0);
	glEnable(GL_CULL_FACE);

	return;
}

void FreeSphereMapData(SPHERE_MAP* pSphereMap)
{
	if (pSphereMap->uiVAO)
	{
		glDeleteVertexArrays(1, &pSphereMap->uiVAO);
		pSphereMap->uiVAO = 0;
	}

	if (pSphereMap->uiVBOPosition)
	{
		glDeleteBuffers(1, &pSphereMap->uiVBOPosition);
		pSphereMap->uiVBOPosition = 0;
	}

	if (pSphereMap->uiVBOTexture)
	{
		glDeleteBuffers(1, &pSphereMap->uiVBOTexture);
		pSphereMap->uiVBOTexture = 0;
	}

	if (pSphereMap->uiVBOElements)
	{
		glDeleteBuffers(1, &pSphereMap->uiVBOElements);
		pSphereMap->uiVBOElements = 0;
	}

	return;
}