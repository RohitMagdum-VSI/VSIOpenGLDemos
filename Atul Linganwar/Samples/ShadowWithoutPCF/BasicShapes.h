#pragma once
#include <Windows.h>

#include <gl\glew.h>
#include <gl\GL.h>

#define GLM_ENABLE_EXPERIMENTAL

#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>

#define PI 3.14152859

enum
{
	GL_ATTRIBUTE_VERTEX = 0,
	GL_ATTRIBUTE_COLOR,
	GL_ATTRIBUTE_NORMAL,
	GL_ATTRIBUTE_TEXTURE0,
	GL_ATTRIBUTE_TEXTURE1,
	GL_ATTRIBUTE_TEXTURE2,
	GL_ATTRIBUTE_TEXTURE3,
};

class CRing
{
private:
	GLuint VAO;
	GLuint VBOPosition;
	GLuint VBONormal;
	GLuint VBOTexture;
	GLuint VBOColor;
	GLuint VBOIndex;

	GLuint iIndicesCount;

public:

	CRing();
	~CRing();

	void InitializeRing(GLfloat InnerRadius, GLfloat OuterRadius, GLint Slices);
	void DrawRing(void);
};

class CSphere
{
private:
	GLuint VAO;
	GLuint VBOPosition;
	GLuint VBONormal;
	GLuint VBOTexture;
	GLuint VBOColor;
	GLuint VBOIndex;

	GLuint iIndicesCount;

public:

	CSphere();
	~CSphere();

	void InitializeSphere(GLfloat Radius, GLint Slices, GLint Stacks);
	void DrawSphere(void);
};

class CTorus
{
private:
	GLuint VAO;
	GLuint VBOPosition;
	GLuint VBONormal;
	GLuint VBOTexture;
	GLuint VBOColor;
	GLuint VBOIndex;

	GLuint iIndicesCount;

public:

	CTorus();
	~CTorus();

	void InitializeTorus(GLfloat InnerRadius, GLfloat OuterRadius, GLint Slices, GLint Stacks);
	void DrawTorus(void);
};

class CCylinder
{
private:
	GLuint VAO;
	GLuint VBOPosition;
	GLuint VBONormal;
	GLuint VBOTexture;
	GLuint VBOColor;
	GLuint VBOIndex;

	GLuint iIndicesCount;

public:

	CCylinder();
	~CCylinder();

	void InitializeCylinder(GLfloat Radius, GLfloat Height, GLint Slices, GLboolean NoTop, GLboolean NoBottom);
	void DrawCylinder(void);
};

class CCone
{
private:
	GLuint VAO;
	GLuint VBOPosition;
	GLuint VBONormal;
	GLuint VBOTexture;
	GLuint VBOColor;
	GLuint VBOIndex;

	GLuint iIndicesCount;

public:

	CCone();
	~CCone();

	void InitializeCone(GLfloat Radius, GLfloat Height, GLint Slices, GLboolean NoBottom);
	void DrawCone(void);
};
