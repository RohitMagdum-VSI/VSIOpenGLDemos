#include<Windows.h>
#include<stdio.h>
#include<stdlib.h>

#include<gl/glew.h>
#include<gl/GL.h>

#include "../common/vmath.h"
#include "VertexData.h"
#include "VAOs.h"
#include "RenderToTexture.h"

VAO::VAO(VAO_TYPE type, SHAPE shape)
{
	m_uiVAO = 0;

	m_uiVBOPosition = 0;
	m_uiVBOColor = 0;
	m_uiVBOTexture = 0;
	m_uiVBONormal = 0;

	m_type = type;
	m_shape = shape;
}

VAO::~VAO()
{

}

bool VAO::Init()
{
	glGenVertexArrays(1, &m_uiVAO);
	glBindVertexArray(m_uiVAO);

	glGenBuffers(1, &m_uiVBOPosition);
	glBindBuffer(GL_ARRAY_BUFFER, m_uiVBOPosition);
	FillVBOPositionData();
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	switch (m_type)
	{
	case VAO_COLOR:
		glGenBuffers(1, &m_uiVBOColor);
		glBindBuffer(GL_ARRAY_BUFFER, m_uiVBOColor);
		FillVBOColorData();
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		break;

	case VAO_TEXTURE:
		glGenBuffers(1, &m_uiVBOTexture);
		glBindBuffer(GL_ARRAY_BUFFER, m_uiVBOTexture);
		FillVBOTextureData();
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		break;

	case VAO_NORMAL:
		break;
	}

	return true;
}

void VAO::DeInit()
{
	if (0 != m_uiVBOPosition)
	{
		glDeleteBuffers(1, &m_uiVBOPosition);
		m_uiVBOPosition = 0;
	}

	if (0 != m_uiVBOColor)
	{
		glDeleteBuffers(1, &m_uiVBOColor);
		m_uiVBOColor = 0;
	}

	if (0 != m_uiVBOTexture)
	{
		glDeleteBuffers(1, &m_uiVBOTexture);
		m_uiVBOTexture = 0;
	}

	if (0 != m_uiVBONormal)
	{
		glDeleteBuffers(1, &m_uiVBONormal);
		m_uiVBONormal = 0;
	}

	if (0 != m_uiVAO)
	{
		glDeleteVertexArrays(1, &m_uiVAO);
		m_uiVAO = 0;
	}

	return;
}

void VAO::FillVBOPositionData(void)
{
	switch (m_shape)
	{
	case QUAD:
		glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
		glVertexAttribPointer(OGL_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(OGL_ATTRIBUTE_POSITION);
		break;

	case PYRAMID:
		glBufferData(GL_ARRAY_BUFFER, sizeof(pyramidVertices), pyramidVertices, GL_STATIC_DRAW);
		glVertexAttribPointer(OGL_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(OGL_ATTRIBUTE_POSITION);
		break;

	case CUBE:
		glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);
		glVertexAttribPointer(OGL_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(OGL_ATTRIBUTE_POSITION);
		break;
	}

	return;
}

void VAO::FillVBOColorData(void)
{
	switch (m_shape)
	{
	case QUAD:
		glBufferData(GL_ARRAY_BUFFER, sizeof(quadColor), quadColor, GL_STATIC_DRAW);
		glVertexAttribPointer(OGL_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(OGL_ATTRIBUTE_COLOR);
		break;

	case PYRAMID:
		glBufferData(GL_ARRAY_BUFFER, sizeof(pyramidColor), pyramidColor, GL_STATIC_DRAW);
		glVertexAttribPointer(OGL_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(OGL_ATTRIBUTE_COLOR);
		break;

	case CUBE:
		glBufferData(GL_ARRAY_BUFFER, sizeof(cubeColor), cubeColor, GL_STATIC_DRAW);
		glVertexAttribPointer(OGL_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(OGL_ATTRIBUTE_COLOR);
		break;
	}

	return;
}

void VAO::FillVBOTextureData(void)
{
	switch (m_shape)
	{
	case QUAD:
		glBufferData(GL_ARRAY_BUFFER, sizeof(quadTexCoords), quadTexCoords, GL_STATIC_DRAW);
		glVertexAttribPointer(OGL_ATTRIBUTE_TEXTURE, 2, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(OGL_ATTRIBUTE_TEXTURE);
		break;

	case PYRAMID:
		glBufferData(GL_ARRAY_BUFFER, sizeof(pyramidTexcoord), pyramidTexcoord, GL_STATIC_DRAW);
		glVertexAttribPointer(OGL_ATTRIBUTE_TEXTURE, 2, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(OGL_ATTRIBUTE_TEXTURE);
		break;

	case CUBE:
		glBufferData(GL_ARRAY_BUFFER, sizeof(cubeTexcoord), cubeTexcoord, GL_STATIC_DRAW);
		glVertexAttribPointer(OGL_ATTRIBUTE_TEXTURE, 2, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(OGL_ATTRIBUTE_TEXTURE);
		break;
	}

	return;
}

void VAO::FillVBONormalData(void)
{
	return;
}
