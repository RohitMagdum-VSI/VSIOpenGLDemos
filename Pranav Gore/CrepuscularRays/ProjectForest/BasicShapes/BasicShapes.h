#pragma once

#include <vector>
#include <Windows.h>

#include <GL/glew.h>
#include <GL/GL.h>

#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>
#include <glm\gtx\string_cast.hpp>

#include "../Data.h"

class CBasicShapes
{
private:
	GLuint m_VBOPosition;
	GLuint m_VBOTexCoord;
	GLuint m_VBONormal;
	GLuint m_VBOTangent;
	GLuint m_VBOColor;

protected:

	GLuint m_VertexCount;
	GLuint m_IndicesCount;

	std::vector<GLuint> m_IndicesBuffer;
	std::vector<glm::vec3> m_VertexBuffer;
	std::vector<glm::vec2> m_TexCoordBuffer;
	std::vector<glm::vec3> m_NormalBuffer;
	std::vector<glm::vec3> m_TangentBuffer;

	void GenerateObjectVAO(GLuint VAO, GLuint EBO, GLboolean ColorOnlyMode);
	void DrawObject(GLuint VAO, GLuint EBO, GLuint IndicesCount);
	void DrawObject(GLuint VAO, GLuint EBO, GLuint IndicesCount, glm::vec4 Color);

public:
	CBasicShapes();
	~CBasicShapes();

};

//////////////////////////////////////////////////

class CRing : protected CBasicShapes
{
private:
	GLfloat m_InnerRadius;
	GLfloat m_OuterRadius;
	GLuint m_Slices;

	GLuint m_VAORing;
	GLuint m_EBORing;
	GLuint m_RingIndicesCount;

protected:

public:
	CRing(GLfloat InnerRadius, GLfloat OuterRadius, GLuint Slices, GLboolean ColorOnlyMode);
	~CRing();

	void DrawRing(void);
	void DrawRing(glm::vec4 Color);
};

/////////////////////////////////////////////////////////////

class CSphere : protected CBasicShapes
{
private:

	GLuint m_VAOSphere;
	GLuint m_EBOSphere;
	GLuint m_SphereIndicesCount;

	GLfloat m_Radius;
	GLuint m_Slices;
	GLuint m_Stacks;

protected:

public:
	CSphere(GLfloat Radius, GLuint Slices, GLuint Stacks, GLboolean ColorOnlyMode);
	~CSphere();

	void DrawSphere(void);
	void DrawSphere(glm::vec4 Color);
};

////////////////////////////////////////////////////
// Torus

class CTorus : protected CBasicShapes
{
private:
	GLuint m_VAOTorus;
	GLuint m_EBOTorus;
	GLuint m_TorusIndicesCount;

	GLfloat m_InnerRadius;
	GLfloat m_OuterRadius;
	GLuint m_Slices;
	GLuint m_Stacks;

protected:

public:
	CTorus(GLfloat InnerRadius, GLfloat OuterRadius, GLint Slices, GLint Stacks, GLboolean ColorOnlyMode);
	~CTorus();

	void DrawTorus(void);
	void DrawTorus(glm::vec4 Color);
};

//////////////////////////////////////////////////////////
// Cylinder

class CCylinder : protected CBasicShapes
{
private:
	GLuint m_VAOCylinder;
	GLuint m_EBOCylinder;
	GLuint m_CylinderIndicesCount;
	GLuint m_TriangleCount;

	GLfloat m_Radius;
	GLfloat m_Height;
	GLuint m_Slices;
	GLboolean m_IsTopPresent;
	GLboolean m_IsBottomPresent;

protected:

public:
	CCylinder(GLfloat Radius, GLfloat Height, GLuint Slices, GLboolean IsTop, GLboolean IsBottom, GLboolean ColorOnlyMode);
	~CCylinder();

	void DrawCylinder(void);
	void DrawCylinder(glm::vec4 Color);
};

////////////////////////////////////////////////////////
// Cone

class CCone : protected CBasicShapes
{
private:
	GLuint m_VAOCone;
	GLuint m_EBOCone;
	GLuint m_ConeIndicesCount;
	GLuint m_TriangleCount;

	GLfloat m_Radius;
	GLfloat m_Height;
	GLuint m_Slices;
	GLboolean m_IsBottomPresent;

protected:

public:
	CCone(GLfloat Radius, GLfloat Height, GLuint Slices, GLboolean IsBottom, GLboolean ColorOnlyMode);
	~CCone();

	void DrawCone(void);
	void DrawCone(glm::vec4 Color);
};
