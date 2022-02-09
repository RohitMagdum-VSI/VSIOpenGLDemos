#include "BasicShapes.h"

/////////////////////////////////////////////////
// Common to all basic shapess

CBasicShapes::CBasicShapes()
{
	// code

	m_VertexBuffer.clear();
	m_TexCoordBuffer.clear();
	m_NormalBuffer.clear();
	m_TangentBuffer.clear();
	m_IndicesBuffer.clear();
}

CBasicShapes::~CBasicShapes()
{
	if (m_VBOPosition)
		glDeleteBuffers(1, &m_VBOPosition);

	if (m_VBONormal)
		glDeleteBuffers(1, &m_VBONormal);

	if (m_VBOTexCoord)
		glDeleteBuffers(1, &m_VBOTexCoord);

	if (m_VBOTexCoord)
		glDeleteBuffers(1, &m_VBOTexCoord);

	m_VertexBuffer.clear();
	m_TexCoordBuffer.clear();
	m_NormalBuffer.clear();
	m_TangentBuffer.clear();
	m_IndicesBuffer.clear();
}

void CBasicShapes::GenerateObjectVAO(GLuint VAO, GLuint EBO)
{
	glBindVertexArray(VAO);

	// VBO Position
	glGenBuffers(1, &m_VBOPosition);
	glBindBuffer(GL_ARRAY_BUFFER, m_VBOPosition);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * m_VertexBuffer.size(), &m_VertexBuffer[0], GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_VERTEX, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_VERTEX);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// VBO Normal
	glGenBuffers(1, &m_VBONormal);
	glBindBuffer(GL_ARRAY_BUFFER, m_VBONormal);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * m_NormalBuffer.size(), &m_NormalBuffer[0], GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_NORMAL);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// VBO Tangent
	glGenBuffers(1, &m_VBOTangent);
	glBindBuffer(GL_ARRAY_BUFFER, m_VBOTangent);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * m_TangentBuffer.size(), &m_TangentBuffer[0], GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_TANGENT, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_TANGENT);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// VBO Texture
	glGenBuffers(1, &m_VBOTexCoord);
	glBindBuffer(GL_ARRAY_BUFFER, m_VBOTexCoord);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * m_TexCoordBuffer.size(), &m_TexCoordBuffer[0], GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_TEXTURE0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * m_IndicesBuffer.size(), &m_IndicesBuffer[0], GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	m_VertexBuffer.clear();
	m_TexCoordBuffer.clear();
	m_NormalBuffer.clear();
	m_TangentBuffer.clear();
	m_IndicesBuffer.clear();
}

void CBasicShapes::GenerateObjectVAO(GLuint VAO, GLuint EBO, glm::vec3 Color)
{
	glBindVertexArray(VAO);

	// VBO Position
	glGenBuffers(1, &m_VBOPosition);
	glBindBuffer(GL_ARRAY_BUFFER, m_VBOPosition);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * m_VertexBuffer.size(), &m_VertexBuffer[0], GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_VERTEX, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_VERTEX);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// VBO Normal
	glGenBuffers(1, &m_VBONormal);
	glBindBuffer(GL_ARRAY_BUFFER, m_VBONormal);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * m_NormalBuffer.size(), &m_NormalBuffer[0], GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_NORMAL);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// VBO Tangent
	glGenBuffers(1, &m_VBOTangent);
	glBindBuffer(GL_ARRAY_BUFFER, m_VBOTangent);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * m_TangentBuffer.size(), &m_TangentBuffer[0], GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_TANGENT, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_TANGENT);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Color the geometry
	glVertexAttrib3fv(OGL_ATTRIBUTE_COLOR, glm::value_ptr(Color));

	// VBO Texture
	glGenBuffers(1, &m_VBOTexCoord);
	glBindBuffer(GL_ARRAY_BUFFER, m_VBOTexCoord);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * m_TexCoordBuffer.size(), &m_TexCoordBuffer[0], GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_TEXTURE0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * m_IndicesBuffer.size(), &m_IndicesBuffer[0], GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	m_VertexBuffer.clear();
	m_TexCoordBuffer.clear();
	m_NormalBuffer.clear();
	m_TangentBuffer.clear();
	m_IndicesBuffer.clear();
}

void CBasicShapes::DrawObject(GLuint VAO, GLuint EBO, GLuint IndicesCount)
{
	glBindVertexArray(VAO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glDrawElements(GL_TRIANGLES, 3 * IndicesCount, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Ring

CRing::CRing(GLfloat InnerRadius, GLfloat OuterRadius, GLuint Slices)
{
	this->m_InnerRadius = InnerRadius;
	this->m_OuterRadius = OuterRadius;
	this->m_Slices = Slices;

	m_VertexCount = (m_InnerRadius == 0) ? m_Slices + 1 : m_Slices * 2;
	m_IndicesCount = (m_InnerRadius == 0) ? m_Slices : 2 * m_Slices;
	m_RingIndicesCount = m_IndicesCount;

	GLfloat d = 2.0f * glm::pi<GLfloat>() / m_Slices;
	GLuint i;

	GLfloat c = 0.0f;
	GLfloat s = 0.0f;

	if (m_InnerRadius == 0)
	{
		for (i = 0; i < m_Slices; i++)
		{
			c = glm::cos(glm::radians(d * i));
			s = glm::sin(glm::radians(d * i));

			m_VertexBuffer.push_back(glm::vec3(c * m_OuterRadius, s * m_OuterRadius, 0.0f));
			m_TexCoordBuffer.push_back(glm::vec2(0.5f + 0.5f * c, 0.5f + 0.5f * s));

			m_IndicesBuffer.push_back(m_Slices);
			m_IndicesBuffer.push_back(i);
			m_IndicesBuffer.push_back(i == m_Slices - 1 ? 0 : i + 1);
		}

		m_VertexBuffer.push_back(glm::vec3(0.0f, 0.0f, 0.0f));
		m_TexCoordBuffer.push_back(glm::vec2(0.0f, 0.0f));
	}
	else
	{
		GLfloat ratio = m_InnerRadius / m_OuterRadius;

		for (i = 0; i < m_Slices; i++)
		{
			c = (GLfloat)glm::cos(d * i);
			s = (GLfloat)glm::sin(d * i);

			m_VertexBuffer.push_back(glm::vec3(c * m_InnerRadius, s * m_InnerRadius, 0.0f));
			m_VertexBuffer.push_back(glm::vec3(c * m_OuterRadius, s * m_OuterRadius, 0.0f));

			m_TexCoordBuffer.push_back(glm::vec2(0.5f + 0.5f * c * ratio, 0.5f + 0.5f * s * ratio));
			m_TexCoordBuffer.push_back(glm::vec2(0.5f + 0.5f * c, 0.5f + 0.5f * s));
		}

		for (i = 0; i < m_Slices - 1; i++)
		{
			m_IndicesBuffer.push_back(2 * i);
			m_IndicesBuffer.push_back(2 * i + 1);
			m_IndicesBuffer.push_back(2 * i + 3);

			m_IndicesBuffer.push_back(2 * i);
			m_IndicesBuffer.push_back(2 * i + 3);
			m_IndicesBuffer.push_back(2 * i + 2);
		}

		m_IndicesBuffer.push_back(2 * i);
		m_IndicesBuffer.push_back(2 * i + 1);
		m_IndicesBuffer.push_back(1);

		m_IndicesBuffer.push_back(2 * i);
		m_IndicesBuffer.push_back(1);
		m_IndicesBuffer.push_back(0);
	}

	// calculate the normals
	for (i = 0; i < m_VertexCount; i++)
	{
		m_NormalBuffer.push_back(glm::vec3(0.0f, 0.0f, 1.0f));
		m_TangentBuffer.push_back(glm::vec3(1.0f, 0.0f, 0.0f));
	}

	glGenVertexArrays(1, &m_VAORing);
	glGenBuffers(1, &m_EBORing);

	GenerateObjectVAO(m_VAORing, m_EBORing);
}

CRing::~CRing()
{
	if (m_VAORing)
		glDeleteVertexArrays(1, &m_VAORing);

	if (m_EBORing)
		glDeleteBuffers(1, &m_EBORing);
}

void CRing::DrawRing(void)
{
	DrawObject(m_VAORing, m_EBORing, m_RingIndicesCount);
}

///////////////////////////////////////////////////////////////////////////////////
// Sphere

CSphere::CSphere(GLfloat Radius, GLuint Slices, GLuint Stacks)
{
	this->m_Radius = Radius;
	this->m_Slices = Slices;
	this->m_Stacks = Stacks;

	m_VertexCount = (m_Slices + 1) * (m_Stacks + 1);
	m_IndicesCount = 2 * m_Slices * m_Stacks * 3;
	m_SphereIndicesCount = m_IndicesCount;

	GLfloat du = 2.0f * glm::pi<GLfloat>() / m_Slices;
	GLfloat dv = glm::pi<GLfloat>() / m_Stacks;

	GLuint i, j;
	GLfloat u, v, x, y, z;

	for (i = 0; i <= m_Stacks; i++)
	{
		v = -glm::pi<GLfloat>() / 2.0f + i * dv;
		for (j = 0; j <= m_Slices; j++)
		{
			u = j * du;

			x = glm::cos(u) * glm::cos(v);
			y = glm::sin(u) * glm::cos(v);
			z = glm::sin(v);

			m_VertexBuffer.push_back(glm::vec3(m_Radius * x, m_Radius * y, m_Radius * z));
			m_NormalBuffer.push_back(glm::vec3(x, y, z));
			m_TangentBuffer.push_back(glm::vec3(-y, x, z));
			m_TexCoordBuffer.push_back(glm::vec2((GLfloat)j / (GLfloat)m_Slices, (GLfloat)i / (GLfloat)m_Stacks));
		}
	}

	for (j = 0; j < m_Stacks; j++)
	{
		GLfloat row1 = j * (m_Slices + 1.0f);
		GLfloat row2 = (j + 1) * (m_Slices + 1);

		for (i = 0; i < m_Slices; i++)
		{
			m_IndicesBuffer.push_back(row1 + i);
			m_IndicesBuffer.push_back(row2 + i + 1);
			m_IndicesBuffer.push_back(row2 + i);

			m_IndicesBuffer.push_back(row1 + i);
			m_IndicesBuffer.push_back(row1 + i + 1);
			m_IndicesBuffer.push_back(row2 + i + 1);
		}
	}

	glGenVertexArrays(1, &m_VAOSphere);
	glGenBuffers(1, &m_EBOSphere);

	GenerateObjectVAO(m_VAOSphere, m_EBOSphere);
}

CSphere::CSphere(GLfloat Radius, GLuint Slices, GLuint Stacks, glm::vec3 Color)
{
	this->m_Radius = Radius;
	this->m_Slices = Slices;
	this->m_Stacks = Stacks;

	m_VertexCount = (m_Slices + 1) * (m_Stacks + 1);
	m_IndicesCount = 2 * m_Slices * m_Stacks * 3;
	m_SphereIndicesCount = m_IndicesCount;

	GLfloat du = 2.0f * glm::pi<GLfloat>() / m_Slices;
	GLfloat dv = glm::pi<GLfloat>() / m_Stacks;

	GLuint i, j;
	GLfloat u, v, x, y, z;

	for (i = 0; i <= m_Stacks; i++)
	{
		v = -glm::pi<GLfloat>() / 2.0f + i * dv;
		for (j = 0; j <= m_Slices; j++)
		{
			u = j * du;

			x = glm::cos(u) * glm::cos(v);
			y = glm::sin(u) * glm::cos(v);
			z = glm::sin(v);

			m_VertexBuffer.push_back(glm::vec3(m_Radius * x, m_Radius * y, m_Radius * z));
			m_NormalBuffer.push_back(glm::vec3(x, y, z));
			m_TangentBuffer.push_back(glm::vec3(-y, x, z));
			m_TexCoordBuffer.push_back(glm::vec2((GLfloat)j / (GLfloat)m_Slices, (GLfloat)i / (GLfloat)m_Stacks));
		}
	}

	for (j = 0; j < m_Stacks; j++)
	{
		GLfloat row1 = j * (m_Slices + 1.0f);
		GLfloat row2 = (j + 1) * (m_Slices + 1);

		for (i = 0; i < m_Slices; i++)
		{
			m_IndicesBuffer.push_back(row1 + i);
			m_IndicesBuffer.push_back(row2 + i + 1);
			m_IndicesBuffer.push_back(row2 + i);

			m_IndicesBuffer.push_back(row1 + i);
			m_IndicesBuffer.push_back(row1 + i + 1);
			m_IndicesBuffer.push_back(row2 + i + 1);
		}
	}

	glGenVertexArrays(1, &m_VAOSphere);
	glGenBuffers(1, &m_EBOSphere);

	GenerateObjectVAO(m_VAOSphere, m_EBOSphere, Color);
}


CSphere::~CSphere()
{
	if (m_VAOSphere)
		glDeleteVertexArrays(1, &m_VAOSphere);

	if (m_EBOSphere)
		glDeleteBuffers(1, &m_EBOSphere);
}

void CSphere::DrawSphere(void)
{
	DrawObject(m_VAOSphere, m_EBOSphere, m_SphereIndicesCount);
}

///////////////////////////////////////////////////////////////
// Torus

CTorus::CTorus(GLfloat InnerRadius, GLfloat OuterRadius, GLint Slices, GLint Stacks)
{
	this->m_InnerRadius = InnerRadius;
	this->m_OuterRadius = OuterRadius;
	this->m_Slices = Slices;
	this->m_Stacks = Stacks;

	m_VertexCount = (m_Slices + 1) * (m_Stacks + 1);
	m_IndicesCount = (2 * m_Slices * m_Stacks * 3);
	m_TorusIndicesCount = m_IndicesCount;

	GLfloat du = 2 * glm::pi<GLfloat>() / m_Slices;
	GLfloat dv = 2 * glm::pi<GLfloat>() / m_Stacks;

	GLfloat CenterRadius = (m_InnerRadius + m_OuterRadius) / 2;
	GLfloat TubeRadius = m_OuterRadius - CenterRadius;

	GLuint i, j;
	GLfloat u, v, cx, cy, SinVal, CosVal, x, y, z;

	for (j = 0; j <= m_Stacks; j++)
	{
		v = -glm::pi<GLfloat>() + j * dv;
		CosVal = glm::cos(v);
		SinVal = glm::sin(v);

		for (i = 0; i <= m_Slices; i++)
		{
			u = i * du;

			cx = glm::cos(u);
			cy = glm::sin(u);

			x = cx * (CenterRadius + TubeRadius * CosVal);
			y = cy * (CenterRadius + TubeRadius * CosVal);
			z = SinVal * TubeRadius;

			m_VertexBuffer.push_back(glm::vec3(x, y, z));
			m_NormalBuffer.push_back(glm::vec3(cx * CosVal, cy * CosVal, SinVal));
			m_TangentBuffer.push_back(glm::vec3(-cy, cx, 0.0f));
			m_TexCoordBuffer.push_back(glm::vec2((GLfloat)i / (GLfloat)m_Slices, (GLfloat)j / (GLfloat)m_Stacks));
		}
	}

	for (j = 0; j < m_Stacks; j++)
	{
		GLfloat row1 = j * (m_Slices + 1);
		GLfloat row2 = (j + 1) * (m_Slices + 1);

		for (i = 0; i < m_Slices; i++)
		{
			m_IndicesBuffer.push_back(row1 + i);
			m_IndicesBuffer.push_back(row2 + i + 1);
			m_IndicesBuffer.push_back(row2 + i);

			m_IndicesBuffer.push_back(row1 + i);
			m_IndicesBuffer.push_back(row1 + i + 1);
			m_IndicesBuffer.push_back(row2 + i + 1);
		}
	}

	glGenVertexArrays(1, &m_VAOTorus);
	glGenBuffers(1, &m_EBOTorus);

	GenerateObjectVAO(m_VAOTorus, m_EBOTorus);
}

CTorus::~CTorus()
{
	if (m_VAOTorus)
		glDeleteVertexArrays(1, &m_VAOTorus);

	if (m_EBOTorus)
		glDeleteBuffers(1, &m_EBOTorus);
}

void CTorus::DrawTorus(void)
{
	DrawObject(m_VAOTorus, m_EBOTorus, m_TorusIndicesCount);
}


///////////////////////////////////////////////////////////////////
// Cylinder

CCylinder::CCylinder(GLfloat Radius, GLfloat Height, GLuint Slices, GLboolean IsTop, GLboolean IsBottom)
{
	this->m_Radius = Radius;
	this->m_Height = Height;
	this->m_Slices = Slices;
	this->m_IsBottomPresent = IsBottom;
	this->m_IsTopPresent = IsTop;

	m_VertexCount = 2 * (m_Slices + 1);

	if (m_IsTopPresent == GL_TRUE)
		m_VertexCount += m_Slices + 2;

	if (m_IsBottomPresent == GL_TRUE)
		m_VertexCount += m_Slices + 2;

	m_TriangleCount = 2 * m_Slices;

	if (m_IsTopPresent == GL_TRUE)
		m_TriangleCount += m_Slices;

	if (m_IsBottomPresent == GL_TRUE)
		m_TriangleCount += m_Slices;

	m_IndicesCount = m_TriangleCount;
	m_CylinderIndicesCount = m_IndicesCount;

	GLfloat du = 2.0f * glm::pi<GLfloat>() / m_Slices;
	GLuint i, StartIndex;
	GLfloat u, CosVal, SinVal;
	GLuint kv = 0;

	for (i = 0; i <= m_Slices; i++)
	{
		u = i * du;
		CosVal = glm::cos(u);
		SinVal = glm::sin(u);

		m_VertexBuffer.push_back(glm::vec3(CosVal * m_Radius, SinVal * m_Radius, -m_Height / 2.0f));
		m_NormalBuffer.push_back(glm::vec3(CosVal, SinVal, 0.0f));
		m_TangentBuffer.push_back(glm::vec3(-SinVal, CosVal, 0.0f));
		m_TexCoordBuffer.push_back(glm::vec2((GLfloat)i / (GLfloat)m_Slices, 0.0f));

		m_VertexBuffer.push_back(glm::vec3(CosVal * m_Radius, SinVal * m_Radius, m_Height / 2.0f));
		m_NormalBuffer.push_back(glm::vec3(CosVal, SinVal, 0.0f));
		m_TangentBuffer.push_back(glm::vec3(-SinVal, CosVal, 0.0f));
		m_TexCoordBuffer.push_back(glm::vec2((GLfloat)i / (GLfloat)m_Slices, 1.0f));

		kv += 6;
	}

	for (i = 0; i < m_Slices; i++)
	{
		m_IndicesBuffer.push_back(2 * i);
		m_IndicesBuffer.push_back(2 * i + 3);
		m_IndicesBuffer.push_back(2 * i + 1);

		m_IndicesBuffer.push_back(2 * i);
		m_IndicesBuffer.push_back(2 * i + 2);
		m_IndicesBuffer.push_back(2 * i + 3);
	}

	//StartIndex = (m_VertexBuffer.size() * 3) / 3;
	StartIndex = kv / 3;

	if (m_IsBottomPresent == GL_TRUE)
	{
		m_VertexBuffer.push_back(glm::vec3(0.0f, 0.0f, -m_Height / 2.0f));
		m_NormalBuffer.push_back(glm::vec3(0.0f, 0.0f, -1.0f));
		m_TangentBuffer.push_back(glm::vec3(-1.0f, 0.0f, 0.0f));
		m_TexCoordBuffer.push_back(glm::vec2(0.5f, 0.5f));

		kv += 3;

		for (i = 0; i <= m_Slices; i++)
		{
			u = 2.0f * glm::pi<GLfloat>() - i * du;
			CosVal = glm::cos(u);
			SinVal = glm::sin(u);

			m_VertexBuffer.push_back(glm::vec3(CosVal * m_Radius, SinVal * m_Radius, -m_Height / 2.0f));
			m_NormalBuffer.push_back(glm::vec3(0.0f, 0.0f, -1.0f));
			m_TangentBuffer.push_back(glm::vec3(-1.0f, 0.0f, 0.0f));
			m_TexCoordBuffer.push_back(glm::vec2(0.5f - 0.5f * CosVal, 0.5f + 0.5f * SinVal));

			kv += 3;
		}

		for (i = 0; i < m_Slices; i++)
		{
			m_IndicesBuffer.push_back(StartIndex);
			m_IndicesBuffer.push_back(StartIndex + i + 1);
			m_IndicesBuffer.push_back(StartIndex + i + 2);
		}
	}

	//StartIndex = (m_VertexBuffer.size() * 3) / 3;
	StartIndex = kv / 3;

	if (m_IsBottomPresent == GL_TRUE)
	{
		m_VertexBuffer.push_back(glm::vec3(0.0f, 0.0f, m_Height / 2.0f));
		m_NormalBuffer.push_back(glm::vec3(0.0f, 0.0f, 1.0f));
		m_TangentBuffer.push_back(glm::vec3(1.0f, 0.0f, 0.0f));
		m_TexCoordBuffer.push_back(glm::vec2(0.5f, 0.5f));

		kv += 3;

		for (i = 0; i <= m_Slices; i++)
		{
			u = i * du;

			CosVal = glm::cos(u);
			SinVal = glm::sin(u);

			m_VertexBuffer.push_back(glm::vec3(CosVal * m_Radius, SinVal * m_Radius, m_Height / 2.0f));
			m_NormalBuffer.push_back(glm::vec3(0.0f, 0.0f, 1.0f));
			m_TangentBuffer.push_back(glm::vec3(1.0f, 0.0f, 0.0f));
			m_TexCoordBuffer.push_back(glm::vec2(0.5f + 0.5f * CosVal, 0.5f + 0.5f * SinVal));

			kv += 3;
		}

		for (i = 0; i < m_Slices; i++)
		{
			m_IndicesBuffer.push_back(StartIndex);
			m_IndicesBuffer.push_back(StartIndex + i + 1);
			m_IndicesBuffer.push_back(StartIndex + i + 2);
		}
	}


	glGenVertexArrays(1, &m_VAOCylinder);
	glGenBuffers(1, &m_EBOCylinder);

	GenerateObjectVAO(m_VAOCylinder, m_EBOCylinder);
}

CCylinder::~CCylinder()
{
	if (m_VAOCylinder)
		glDeleteVertexArrays(1, &m_VAOCylinder);

	if (m_EBOCylinder)
		glDeleteBuffers(1, &m_EBOCylinder);
}

void CCylinder::DrawCylinder(void)
{
	DrawObject(m_VAOCylinder, m_EBOCylinder, m_CylinderIndicesCount);
}

///////////////////////////////////////////////////////////////////////////////
// Cone

CCone::CCone(GLfloat Radius, GLfloat Height, GLuint Slices, GLboolean IsBottom)
{
	this->m_Radius = Radius;
	this->m_Height = Height;
	this->m_Slices = Slices;
	this->m_IsBottomPresent = IsBottom;

	GLfloat Fractions[] = { 0.0f, 0.5f, 0.75f, 0.875f, 0.9375f };
	GLuint Length = sizeof(Fractions) / sizeof(GLfloat);

	m_VertexCount = Length * (m_Slices + 1) + m_Slices;

	if (m_IsBottomPresent == GL_TRUE)
		m_VertexCount += m_Slices + 2;

	m_TriangleCount = (Length - 1) * m_Slices * 2 + m_Slices;

	if (m_IsBottomPresent == GL_TRUE)
		m_TriangleCount += m_Slices;

	m_IndicesCount = m_TriangleCount;
	m_ConeIndicesCount = m_IndicesCount;

	GLfloat NormalLength = glm::sqrt(m_Height * m_Height + m_Radius * m_Radius);
	GLfloat n1 = m_Height / NormalLength;
	GLfloat n2 = m_Radius / NormalLength;
	GLfloat du = 2.0f * glm::pi<GLfloat>() / m_Slices;
	GLfloat u, CosVal, SinVal;
	GLuint StartIndex, i, j, kv = 0;

	for (j = 0; j < Length; j++)
	{
		GLfloat Offset = (j % 2 == 0 ? 0 : 0.5f);

		for (i = 0; i <= m_Slices; i++)
		{
			GLfloat h1 = -m_Height / 2.0f + Fractions[j] * m_Height;
			u = (i + Offset) * du;

			CosVal = glm::cos(u);
			SinVal = glm::sin(u);

			m_VertexBuffer.push_back(glm::vec3(CosVal * m_Radius * (1 - Fractions[j]), SinVal * m_Radius * (1 - Fractions[j]), h1));
			m_NormalBuffer.push_back(glm::vec3(CosVal * n1, SinVal * n1, n2));
			m_TangentBuffer.push_back(glm::vec3(-SinVal, CosVal, n2));
			m_TexCoordBuffer.push_back(glm::vec2(((GLfloat)i + (GLfloat)Offset) / (GLfloat)m_Slices, Fractions[j]));

			kv += 3;
		}
	}

	for (j = 0; j < Length - 1; j++)
	{
		GLfloat row1 = j * (m_Slices + 1);
		GLfloat row2 = (j + 1) * (m_Slices + 1);

		for (i = 0; i < m_Slices; i++)
		{
			m_IndicesBuffer.push_back(row1 + i);
			m_IndicesBuffer.push_back(row2 + i + 1);
			m_IndicesBuffer.push_back(row2 + i);

			m_IndicesBuffer.push_back(row1 + i);
			m_IndicesBuffer.push_back(row1 + i + 1);
			m_IndicesBuffer.push_back(row2 + i + 1);
		}
	}

	StartIndex = kv / 3 - (m_Slices + 1);

	for (i = 0; i < m_Slices; i++)
	{
		u = (i + 0.5f) * du;
		CosVal = glm::cos(u);
		SinVal = glm::sin(u);

		m_VertexBuffer.push_back(glm::vec3(0.0f, 0.0f, m_Height / 2.0f));
		m_NormalBuffer.push_back(glm::vec3(CosVal * n1, SinVal * n1, n2));
		m_TangentBuffer.push_back(glm::vec3(-SinVal, CosVal, n2));
		m_TexCoordBuffer.push_back(glm::vec2(((GLfloat)i + 0.5f) / (GLfloat)m_Slices, 1.0f));

		kv += 3;
	}

	for (i = 0; i < m_Slices; i++)
	{
		m_IndicesBuffer.push_back(StartIndex + i);
		m_IndicesBuffer.push_back(StartIndex + i + 1);
		m_IndicesBuffer.push_back(StartIndex + (m_Slices + 1) + i);
	}

	if (m_IsBottomPresent == GL_TRUE)
	{
		StartIndex = kv / 3;

		m_VertexBuffer.push_back(glm::vec3(0.0f, 0.0f, -m_Height / 2.0f));
		m_NormalBuffer.push_back(glm::vec3(0.0f, 0.0f, -1.0f));
		m_TangentBuffer.push_back(glm::vec3(-1.0f, 0.0f, 0.0f));
		m_TexCoordBuffer.push_back(glm::vec2(0.5f, 0.5f));

		kv += 3;

		for (i = 0; i <= m_Slices; i++)
		{
			u = 2.0f * glm::pi<GLfloat>() - i * du;
			CosVal = glm::cos(u);
			SinVal = glm::sin(u);

			m_VertexBuffer.push_back(glm::vec3(CosVal * m_Radius, SinVal * m_Radius, -m_Height / 2.0f));
			m_NormalBuffer.push_back(glm::vec3(0.0f, 0.0f, -1.0f));
			m_TangentBuffer.push_back(glm::vec3(-1.0f, 0.0f, 0.0f));
			m_TexCoordBuffer.push_back(glm::vec2(0.5f - 0.5f * CosVal, 0.5f + 0.5f * SinVal));
		}

		for (i = 0; i < m_Slices; i++)
		{
			m_IndicesBuffer.push_back(StartIndex);
			m_IndicesBuffer.push_back(StartIndex + i + 1);
			m_IndicesBuffer.push_back(StartIndex + i + 2);
		}
	}

	glGenVertexArrays(1, &m_VAOCone);
	glGenBuffers(1, &m_EBOCone);

	GenerateObjectVAO(m_VAOCone, m_EBOCone);
}

CCone::~CCone()
{
	if (m_VAOCone)
		glDeleteVertexArrays(1, &m_VAOCone);

	if (m_EBOCone)
		glDeleteBuffers(1, &m_EBOCone);
}

void CCone::DrawCone(void)
{
	DrawObject(m_VAOCone, m_EBOCone, m_ConeIndicesCount);
}