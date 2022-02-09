#include <stdio.h>
#include "BasicShapes.h"

//extern FILE *gpFile;

CRing::CRing()
{
	// code
	iIndicesCount = 0;
}

CRing::~CRing()
{
	// code
}

//Ring Or Disk
void CRing::InitializeRing(GLfloat fInnerRadius, GLfloat fOuterRadius, GLint iSlices)
{
	GLint iVertexCountRing;

	iVertexCountRing = (fInnerRadius == 0) ? iSlices + 1 : iSlices * 2;

	GLfloat *pfVerticesRing, *pfNormalsRing, *pfTexCoordsRing;
	GLuint *piIndicesRing;
	GLfloat c, s;

	GLdouble d = 2 * PI / iSlices;
	GLint k = 0, t = 0, n = 0, i = 0;

	pfVerticesRing = (GLfloat *)calloc(3 * iVertexCountRing, sizeof(GLfloat));
	pfNormalsRing = (GLfloat *)calloc(3 * iVertexCountRing, sizeof(GLfloat));
	pfTexCoordsRing = (GLfloat *)calloc(2 * iVertexCountRing, sizeof(GLfloat));

	piIndicesRing = (GLuint *)calloc(fInnerRadius == 0 ? 3 * iSlices : 3 * 2 * iSlices, sizeof(GLuint));

	if (fInnerRadius == 0) 
	{
		for (i = 0; i < iSlices; i++)
		{
			c = (GLfloat)cos(d * i);
			s = (GLfloat)sin(d * i);
			
			pfVerticesRing[k++] = c * fOuterRadius;
			pfVerticesRing[k++] = s * fOuterRadius;
			pfVerticesRing[k++] = 0;
			
			pfTexCoordsRing[t++] = (GLfloat)(0.5 + 0.5 * c);
			pfTexCoordsRing[t++] = (GLfloat)(0.5 + 0.5 * s);
			
			piIndicesRing[n++] = iSlices;
			piIndicesRing[n++] = i;
			piIndicesRing[n++] = i == iSlices - 1 ? 0 : i + 1;
		}

		pfVerticesRing[k++] = pfVerticesRing[k++] = pfVerticesRing[k++] = 0;
		pfTexCoordsRing[t++] = pfTexCoordsRing[t++] = 0;
	}
	else 
	{
		GLfloat r = fInnerRadius / fOuterRadius;

		for (i = 0; i < iSlices; i++) 
		{
			c = (GLfloat)cos(d * i);
			s = (GLfloat)sin(d * i);

			pfVerticesRing[k++] = c * fInnerRadius;
			pfVerticesRing[k++] = s * fInnerRadius;
			pfVerticesRing[k++] = 0;
			
			pfTexCoordsRing[t++] = (GLfloat)(0.5 + 0.5 * c * r);
			pfTexCoordsRing[t++] = (GLfloat)(0.5 + 0.5 * s * r);
			
			pfVerticesRing[k++] = c * fOuterRadius;
			pfVerticesRing[k++] = s * fOuterRadius;
			pfVerticesRing[k++] = 0;
			
			pfTexCoordsRing[t++] = (GLfloat)(0.5 + 0.5 * c);
			pfTexCoordsRing[t++] = (GLfloat)(0.5 + 0.5 * s);
		}

		for (i = 0; i < iSlices - 1; i++) 
		{
			piIndicesRing[n++] = 2 * i;
			piIndicesRing[n++] = 2 * i + 1;
			piIndicesRing[n++] = 2 * i + 3;
			piIndicesRing[n++] = 2 * i;
			piIndicesRing[n++] = 2 * i + 3;
			piIndicesRing[n++] = 2 * i + 2;
		}

		piIndicesRing[n++] = 2 * i;
		piIndicesRing[n++] = 2 * i + 1;
		piIndicesRing[n++] = 1;
		piIndicesRing[n++] = 2 * i;
		piIndicesRing[n++] = 1;
		piIndicesRing[n++] = 0;
	}

	for (i = 0; i < iVertexCountRing; i++) 
	{
		pfNormalsRing[3 * i] = pfNormalsRing[3 * i + 1] = 0;
		pfNormalsRing[3 * i + 2] = 1;
	}

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	glGenBuffers(1, &VBOPosition);
	glBindBuffer(GL_ARRAY_BUFFER, VBOPosition);
	glBufferData(GL_ARRAY_BUFFER, 3 * iVertexCountRing * sizeof(GLfloat *), pfVerticesRing, GL_STATIC_DRAW);

	glVertexAttribPointer(GL_ATTRIBUTE_VERTEX, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(GL_ATTRIBUTE_VERTEX);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &VBONormal);
	glBindBuffer(GL_ARRAY_BUFFER, VBONormal);
	glBufferData(GL_ARRAY_BUFFER, 3 * iVertexCountRing * sizeof(GLfloat *), pfNormalsRing, GL_STATIC_DRAW);

	glVertexAttribPointer(GL_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(GL_ATTRIBUTE_NORMAL);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &VBOTexture);
	glBindBuffer(GL_ARRAY_BUFFER, VBOTexture);
	glBufferData(GL_ARRAY_BUFFER, 2 * iVertexCountRing * sizeof(GLfloat *), pfTexCoordsRing, GL_STATIC_DRAW);

	glVertexAttribPointer(GL_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(GL_ATTRIBUTE_TEXTURE0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &VBOIndex);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBOIndex);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, (fInnerRadius == 0 ? 3 * iSlices : 3 * 2 * iSlices) * sizeof(GLuint *), piIndicesRing, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	//return
	iIndicesCount = fInnerRadius == 0 ? 3 * iSlices : 3 * 2 * iSlices;

	free(pfVerticesRing);
	free(pfNormalsRing);
	free(pfTexCoordsRing);
	free(piIndicesRing);
}

void CRing::DrawRing(void)
{
	//bind Vao
	glBindVertexArray(VAO);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBOIndex);
	glDrawElements(GL_TRIANGLES, iIndicesCount, GL_UNSIGNED_INT, 0);

	//unbind Vao
	glBindVertexArray(0);
}

CSphere::CSphere()
{
	// code
	iIndicesCount = 0;
}

CSphere::~CSphere()
{
	// code
}

//Sphere
void CSphere::InitializeSphere(GLfloat fRadius, GLint iSlices, GLint iStacks)
{
	GLint iVertexCountSphere = (iSlices + 1) * (iStacks + 1);

	GLfloat *pfVerticesSphere, *pfNormalsSphere, *pfTexCoordsSphere;
	GLuint *piIndicesSphere;

	GLfloat du = (GLfloat)(2 * PI / iSlices);
	GLfloat dv = (GLfloat)(PI / iStacks);
	GLfloat u, v, x, y, z;
	GLint i, j, k = 0;
	GLint IndexV = 0;
	GLint IndexT = 0;

	pfVerticesSphere = (GLfloat *)calloc(3 * iVertexCountSphere, sizeof(GLfloat));
	pfNormalsSphere = (GLfloat *)calloc(3 * iVertexCountSphere, sizeof(GLfloat));
	pfTexCoordsSphere = (GLfloat *)calloc(2 * iVertexCountSphere, sizeof(GLfloat));

	piIndicesSphere = (GLuint *)calloc(2 * iSlices * iStacks * 3, sizeof(GLuint));

	for (i = 0; i <= iStacks; i++)
	{
		v = (GLfloat)(-PI / 2 + i * dv);
		for (j = 0; j <= iSlices; j++)
		{
			u = j * du;
			x = cos(u) * cos(v);
			y = sin(u) * cos(v);
			z = sin(v);

			pfVerticesSphere[IndexV] = fRadius * x;
			pfNormalsSphere[IndexV++] = x;

			pfVerticesSphere[IndexV] = fRadius * y;
			pfNormalsSphere[IndexV++] = y;

			pfVerticesSphere[IndexV] = fRadius * z;
			pfNormalsSphere[IndexV++] = z;

			pfTexCoordsSphere[IndexT++] = (GLfloat)j / (GLfloat)iSlices;
			pfTexCoordsSphere[IndexT++] = (GLfloat)i / (GLfloat)iStacks;
		}
	}

	for (j = 0; j < iStacks; j++)
	{
		GLuint row1 = (GLuint)(j * (iSlices + 1));
		GLuint row2 = (GLuint)((j + 1) * (iSlices + 1));

		for (i = 0; i < iSlices; i++)
		{
			piIndicesSphere[k++] = row1 + i;
			piIndicesSphere[k++] = row2 + i + 1;
			piIndicesSphere[k++] = row2 + i;
			piIndicesSphere[k++] = row1 + i;
			piIndicesSphere[k++] = row1 + i + 1;
			piIndicesSphere[k++] = row2 + i + 1;
		}
	}

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	glGenBuffers(1, &VBOPosition);
	glBindBuffer(GL_ARRAY_BUFFER, VBOPosition);
	glBufferData(GL_ARRAY_BUFFER, 3 * iVertexCountSphere * sizeof(GLfloat *), pfVerticesSphere, GL_STATIC_DRAW);

	glVertexAttribPointer(GL_ATTRIBUTE_VERTEX, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(GL_ATTRIBUTE_VERTEX);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &VBONormal);
	glBindBuffer(GL_ARRAY_BUFFER, VBONormal);
	glBufferData(GL_ARRAY_BUFFER, 3 * iVertexCountSphere * sizeof(GLfloat *), pfNormalsSphere, GL_STATIC_DRAW);

	glVertexAttribPointer(GL_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(GL_ATTRIBUTE_NORMAL);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &VBOTexture);
	glBindBuffer(GL_ARRAY_BUFFER, VBOTexture);
	glBufferData(GL_ARRAY_BUFFER, 2 * iVertexCountSphere * sizeof(GLfloat *), pfTexCoordsSphere, GL_STATIC_DRAW);

	glVertexAttribPointer(GL_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(GL_ATTRIBUTE_TEXTURE0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &VBOIndex);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBOIndex);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, 2 * iSlices * iStacks * 3 * sizeof(GLuint *), piIndicesSphere, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	//return
	iIndicesCount = 2 * iSlices * iStacks * 3;

	free(pfVerticesSphere);
	free(pfNormalsSphere);
	free(pfTexCoordsSphere);
	free(piIndicesSphere);
}

void CSphere::DrawSphere(void)
{
	//bind Vao
	glBindVertexArray(VAO);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBOIndex);
	glDrawElements(GL_TRIANGLES, iIndicesCount, GL_UNSIGNED_INT, 0);

	//unbind Vao
	glBindVertexArray(0);
}

CTorus::CTorus()
{
	// code
	iIndicesCount = 0;
}

CTorus::~CTorus()
{
	// code
}

//Torus
void CTorus::InitializeTorus(GLfloat fInnerRadius, GLfloat fOuterRadius, GLint iSlices, GLint iStacks)
{
	GLint iVertexCountTorus = (iSlices + 1) * (iStacks + 1);

	GLfloat *pfVerticesTorus, *pfNormalsTorus, *pfTexCoordsTorus;
	GLuint *piIndicesTorus;

	pfVerticesTorus = (GLfloat *)calloc(3 * iVertexCountTorus, sizeof(GLfloat));
	pfNormalsTorus = (GLfloat *)calloc(3 * iVertexCountTorus, sizeof(GLfloat));
	pfTexCoordsTorus = (GLfloat *)calloc(2 * iVertexCountTorus, sizeof(GLfloat));

	piIndicesTorus = (GLuint *)calloc(2 * iSlices * iStacks * 3, sizeof(GLuint));

	GLfloat du = (GLfloat)(2 * PI / iSlices);
	GLfloat dv = (GLfloat)(2 * PI / iStacks);

	GLfloat fCenterRadius = (fInnerRadius + fOuterRadius) / 2;
	GLfloat fTubeRadius = fOuterRadius - fCenterRadius;

	GLint i, j, k = 0;
	GLfloat u, v, c, s, cx, cy, x, y, z;
	GLint IndexV = 0;
	GLint IndexT = 0;

	for (j = 0; j <= iStacks; j++)
	{
		v = (GLfloat)(-PI + j * dv);
		c = cos(v);
		s = sin(v);

		for (i = 0; i <= iSlices; i++)
		{
			u = i * du;
			cx = cos(u);
			cy = sin(u);

			x = cx * (fCenterRadius + fTubeRadius * c);
			y = cy * (fCenterRadius + fTubeRadius * c);
			z = s * fTubeRadius;

			pfVerticesTorus[IndexV] = x;
			pfNormalsTorus[IndexV++] = cx * c;

			pfVerticesTorus[IndexV] = y;
			pfNormalsTorus[IndexV++] = cy * c;
			
			pfVerticesTorus[IndexV] = z;
			pfNormalsTorus[IndexV++] = s;

			pfTexCoordsTorus[IndexT++] = (GLfloat)i / (GLfloat)iSlices;
			pfTexCoordsTorus[IndexT++] = (GLfloat)j / (GLfloat)iStacks;
		}
	}

	for (j = 0; j < iStacks; j++)
	{
		GLuint row1 = j * (iSlices + 1);
		GLuint row2 = (j + 1) * (iSlices + 1);

		for (i = 0; i < iSlices; i++)
		{
			piIndicesTorus[k++] = row1 + i;
			piIndicesTorus[k++] = row2 + i + 1;
			piIndicesTorus[k++] = row2 + i;
			piIndicesTorus[k++] = row1 + i;
			piIndicesTorus[k++] = row1 + i + 1;
			piIndicesTorus[k++] = row2 + i + 1;
		}
	}

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	glGenBuffers(1, &VBOPosition);
	glBindBuffer(GL_ARRAY_BUFFER, VBOPosition);
	glBufferData(GL_ARRAY_BUFFER, 3 * iVertexCountTorus * sizeof(GLfloat *), pfVerticesTorus, GL_STATIC_DRAW);

	glVertexAttribPointer(GL_ATTRIBUTE_VERTEX, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(GL_ATTRIBUTE_VERTEX);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &VBONormal);
	glBindBuffer(GL_ARRAY_BUFFER, VBONormal);
	glBufferData(GL_ARRAY_BUFFER, 3 * iVertexCountTorus * sizeof(GLfloat *), pfNormalsTorus, GL_STATIC_DRAW);

	glVertexAttribPointer(GL_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(GL_ATTRIBUTE_NORMAL);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &VBOTexture);
	glBindBuffer(GL_ARRAY_BUFFER, VBOTexture);
	glBufferData(GL_ARRAY_BUFFER, 2 * iVertexCountTorus * sizeof(GLfloat *), pfTexCoordsTorus, GL_STATIC_DRAW);

	glVertexAttribPointer(GL_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(GL_ATTRIBUTE_TEXTURE0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &VBOIndex);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBOIndex);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, 2 * iSlices * iStacks * 3 * sizeof(GLuint *), piIndicesTorus, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	//return
	iIndicesCount = 2 * iSlices * iStacks * 3;

	free(pfVerticesTorus);
	free(pfNormalsTorus);
	free(pfTexCoordsTorus);
	free(piIndicesTorus);
}

void CTorus::DrawTorus(void)
{
	//bind Vao
	glBindVertexArray(VAO);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBOIndex);
	glDrawElements(GL_TRIANGLES, iIndicesCount, GL_UNSIGNED_INT, 0);

	//unbind Vao
	glBindVertexArray(0);
}

CCylinder::CCylinder()
{
	// code
	iIndicesCount = 0;
}

CCylinder::~CCylinder()
{
	// code
}

//Cylinder
void CCylinder::InitializeCylinder(GLfloat fRadius, GLfloat fHeight, GLint iSlices, GLboolean bNoTop, GLboolean bNoBottom)
{
	GLint iVertexCountCylinder = 2 * (iSlices + 1);
	if (!bNoTop)
	{
		iVertexCountCylinder += iSlices + 2;
	}
		
	if (!bNoBottom)
	{
		iVertexCountCylinder += iSlices + 2;
	}

	GLuint iTriangleCountCylinder = 2 * iSlices;

	if (!bNoTop)
	{
		iTriangleCountCylinder += iSlices;
	}
		
	if (!bNoBottom)
	{
		iTriangleCountCylinder += iSlices;
	}

	GLfloat *pfVerticesCylinder, *pfNormalsCylinder, *pfTexCoordsCylinder;
	GLuint *piIndicesCylinder;

	pfVerticesCylinder = (GLfloat *)calloc(3 * iVertexCountCylinder, sizeof(GLfloat));
	pfNormalsCylinder = (GLfloat *)calloc(3 * iVertexCountCylinder, sizeof(GLfloat));
	pfTexCoordsCylinder = (GLfloat *)calloc(2 * iVertexCountCylinder, sizeof(GLfloat));

	piIndicesCylinder = (GLuint *)calloc(iTriangleCountCylinder * 3, sizeof(GLuint));

	GLfloat du = (GLfloat)(2 * PI / iSlices);
	GLint kv = 0;
	GLint kt = 0;
	GLint i, k = 0;
	GLfloat u;

	for (i = 0; i <= iSlices; i++)
	{
		u = i * du;
		GLfloat c = cos(u);
		GLfloat s = sin(u);
		
		pfVerticesCylinder[kv] = c * fRadius;
		pfNormalsCylinder[kv++] = c;

		pfVerticesCylinder[kv] = s * fRadius;
		pfNormalsCylinder[kv++] = s;

		pfVerticesCylinder[kv] = -fHeight / 2;
		pfNormalsCylinder[kv++] = 0;
		
		pfTexCoordsCylinder[kt++] = (GLfloat)i / (GLfloat)iSlices;
		pfTexCoordsCylinder[kt++] = 0;
		
		pfVerticesCylinder[kv] = c * fRadius;
		pfNormalsCylinder[kv++] = c;
		
		pfVerticesCylinder[kv] = s * fRadius;
		pfNormalsCylinder[kv++] = s;

		pfVerticesCylinder[kv] = fHeight / 2;
		pfNormalsCylinder[kv++] = 0;
		
		pfTexCoordsCylinder[kt++] = (GLfloat)i / (GLfloat)iSlices;
		pfTexCoordsCylinder[kt++] = 1;
	}

	for (i = 0; i < iSlices; i++)
	{
		piIndicesCylinder[k++] = 2 * i;
		piIndicesCylinder[k++] = 2 * i + 3;
		piIndicesCylinder[k++] = 2 * i + 1;
		piIndicesCylinder[k++] = 2 * i;
		piIndicesCylinder[k++] = 2 * i + 2;
		piIndicesCylinder[k++] = 2 * i + 3;
	}

	GLint iStartIndex = kv / 3;

	if (!bNoBottom)
	{
		pfVerticesCylinder[kv] = 0;
		pfNormalsCylinder[kv++] = 0;
		
		pfVerticesCylinder[kv] = 0;
		pfNormalsCylinder[kv++] = 0;
		
		pfVerticesCylinder[kv] = -fHeight / 2;
		pfNormalsCylinder[kv++] = -1;

		pfTexCoordsCylinder[kt++] = 0.5;
		pfTexCoordsCylinder[kt++] = 0.5;
		
		for (i = 0; i <= iSlices; i++)
		{
			u = (GLfloat)(2 * PI - i * du);
			GLfloat c = cos(u);
			GLfloat s = sin(u);

			pfVerticesCylinder[kv] = c * fRadius;
			pfNormalsCylinder[kv++] = 0;
			
			pfVerticesCylinder[kv] = s * fRadius;
			pfNormalsCylinder[kv++] = 0;
			
			pfVerticesCylinder[kv] = -fHeight / 2;
			pfNormalsCylinder[kv++] = -1;
			
			pfTexCoordsCylinder[kt++] = (GLfloat)(0.5 - 0.5*c);
			pfTexCoordsCylinder[kt++] = (GLfloat)(0.5 + 0.5*s);
		}

		for (i = 0; i < iSlices; i++)
		{
			piIndicesCylinder[k++] = iStartIndex;
			piIndicesCylinder[k++] = iStartIndex + i + 1;
			piIndicesCylinder[k++] = iStartIndex + i + 2;
		}
	}
	iStartIndex = kv / 3;

	if (!bNoTop)
	{
		pfVerticesCylinder[kv] = 0;
		pfNormalsCylinder[kv++] = 0;
		
		pfVerticesCylinder[kv] = 0;
		pfNormalsCylinder[kv++] = 0;
		
		pfVerticesCylinder[kv] = fHeight / 2;
		pfNormalsCylinder[kv++] = 1;

		pfTexCoordsCylinder[kt++] = 0.5;
		pfTexCoordsCylinder[kt++] = 0.5;
		
		for (i = 0; i <= iSlices; i++)
		{
			u = i * du;
			GLfloat c = cos(u);
			GLfloat s = sin(u);

			pfVerticesCylinder[kv] = c * fRadius;
			pfNormalsCylinder[kv++] = 0;
			
			pfVerticesCylinder[kv] = s * fRadius;
			pfNormalsCylinder[kv++] = 0;
			
			pfVerticesCylinder[kv] = fHeight / 2;
			pfNormalsCylinder[kv++] = 1;
			
			pfTexCoordsCylinder[kt++] = (GLfloat)(0.5 + 0.5 * c);
			pfTexCoordsCylinder[kt++] = (GLfloat)(0.5 + 0.5 * s);
		}

		for (i = 0; i < iSlices; i++)
		{
			piIndicesCylinder[k++] = iStartIndex;
			piIndicesCylinder[k++] = iStartIndex + i + 1;
			piIndicesCylinder[k++] = iStartIndex + i + 2;
		}
	}

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	glGenBuffers(1, &VBOPosition);
	glBindBuffer(GL_ARRAY_BUFFER, VBOPosition);
	glBufferData(GL_ARRAY_BUFFER, 3 * iVertexCountCylinder * sizeof(GLfloat *), pfVerticesCylinder, GL_STATIC_DRAW);

	glVertexAttribPointer(GL_ATTRIBUTE_VERTEX, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(GL_ATTRIBUTE_VERTEX);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &VBONormal);
	glBindBuffer(GL_ARRAY_BUFFER, VBONormal);
	glBufferData(GL_ARRAY_BUFFER, 3 * iVertexCountCylinder * sizeof(GLfloat *), pfNormalsCylinder, GL_STATIC_DRAW);

	glVertexAttribPointer(GL_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(GL_ATTRIBUTE_NORMAL);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &VBOTexture);
	glBindBuffer(GL_ARRAY_BUFFER, VBOTexture);
	glBufferData(GL_ARRAY_BUFFER, 2 * iVertexCountCylinder * sizeof(GLfloat *), pfTexCoordsCylinder, GL_STATIC_DRAW);

	glVertexAttribPointer(GL_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(GL_ATTRIBUTE_TEXTURE0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &VBOIndex);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBOIndex);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, iTriangleCountCylinder * 3 * sizeof(GLuint *), piIndicesCylinder, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	//return
	iIndicesCount = iTriangleCountCylinder * 3;

	free(pfVerticesCylinder);
	free(pfNormalsCylinder);
	free(pfTexCoordsCylinder);
	free(piIndicesCylinder);
}

void CCylinder::DrawCylinder(void)
{
	//bind Vao
	glBindVertexArray(VAO);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBOIndex);
	glDrawElements(GL_TRIANGLES, iIndicesCount, GL_UNSIGNED_INT, 0);

	//unbind Vao
	glBindVertexArray(0);
}



CCone::CCone()
{
	// code
	iIndicesCount = 0;
}

CCone::~CCone()
{
	// code
}

//Cone
void CCone::InitializeCone(GLfloat fRadius, GLfloat fHeight, GLint iSlices, GLboolean bNoBottom)
{
	GLfloat Fractions[] = { 0.0f, 0.5f, 0.75f, 0.875f, 0.9375f };

	GLint iVertexCountCone = 5 * (iSlices + 1) + iSlices;

	if (!bNoBottom)
	{
		iVertexCountCone += iSlices + 2;
	}
		
	GLint iTriangleCountCone = (5 - 1) * iSlices * 2 + iSlices;

	if (!bNoBottom)
	{
		iTriangleCountCone += iSlices;
	}
	
	GLfloat *pfVerticesCone, *pfNormalsCone, *pfTexCoordsCone;
	GLuint *piIndicesCone;

	pfVerticesCone = (GLfloat *)calloc(3 * iVertexCountCone, sizeof(GLfloat));
	pfNormalsCone = (GLfloat *)calloc(3 * iVertexCountCone, sizeof(GLfloat));
	pfTexCoordsCone = (GLfloat *)calloc(2 * iVertexCountCone, sizeof(GLfloat));

	piIndicesCone = (GLuint *)calloc(iTriangleCountCone * 3, sizeof(GLuint));

	GLfloat fNormalLength = sqrtf((fHeight * fHeight) + (fRadius * fRadius));

	GLfloat n1 = fHeight / fNormalLength;
	GLfloat n2 = fRadius / fNormalLength;
	GLfloat du = (GLfloat)(2 * PI / iSlices);

	GLint kv = 0;
	GLint kt = 0;
	GLint i, j, k = 0;
	GLfloat u;

	for (j = 0; j < 5; j++) 
	{
		GLfloat uoffset = (GLfloat)((j % 2 == 0 ? 0 : 0.5));
		
		for (i = 0; i <= iSlices; i++) 
		{
			GLfloat h1 = -fHeight / 2 + Fractions[j] * fHeight;

			u = (i + uoffset) * du;
			
			GLfloat c = cos(u);
			GLfloat s = sin(u);

			pfVerticesCone[kv] = c * fRadius * (1 - Fractions[j]);
			pfNormalsCone[kv++] = c * n1;
			
			pfVerticesCone[kv] = s * fRadius * (1 - Fractions[j]);
			pfNormalsCone[kv++] = s*n1;

			pfVerticesCone[kv] = h1;
			pfNormalsCone[kv++] = n2;
			
			pfTexCoordsCone[kt++] = (i + uoffset) / iSlices;
			pfTexCoordsCone[kt++] = Fractions[j];
		}
	}

	k = 0;
	
	for (j = 0; j < 5 - 1; j++) 
	{
		GLint row1 = j * (iSlices + 1);
		GLint row2 = (j + 1) * (iSlices + 1);

		for (i = 0; i < iSlices; i++) 
		{
			piIndicesCone[k++] = row1 + i;
			piIndicesCone[k++] = row2 + i + 1;
			piIndicesCone[k++] = row2 + i;
			piIndicesCone[k++] = row1 + i;
			piIndicesCone[k++] = row1 + i + 1;
			piIndicesCone[k++] = row2 + i + 1;
		}
	}

	GLint start = kv / 3 - (iSlices + 1);
	
	for (i = 0; i < iSlices; i++) // slices points at top, with different normals, texcoords
	{ 
		u = (GLfloat)((i + 0.5) * du);
		GLfloat c = cos(u);
		GLfloat s = sin(u);

		pfVerticesCone[kv] = 0;
		pfNormalsCone[kv++] = c * n1;

		pfVerticesCone[kv] = 0;
		pfNormalsCone[kv++] = s * n1;

		pfVerticesCone[kv] = fHeight / 2;
		pfNormalsCone[kv++] = n2;
		
		pfTexCoordsCone[kt++] = (GLfloat)((i + 0.5) / iSlices);
		pfTexCoordsCone[kt++] = 1;
	}

	for (i = 0; i < iSlices; i++) 
	{
		piIndicesCone[k++] = start + i;
		piIndicesCone[k++] = start + i + 1;
		piIndicesCone[k++] = start + (iSlices + 1) + i;
	}

	if (!bNoBottom) 
	{
		GLint startIndex = kv / 3;

		pfVerticesCone[kv] = 0;
		pfNormalsCone[kv++] = 0;
		
		pfVerticesCone[kv] = 0;
		pfNormalsCone[kv++] = 0;
		
		pfVerticesCone[kv] = -fHeight / 2;
		pfNormalsCone[kv++] = -1;
		
		pfTexCoordsCone[kt++] = 0.5;
		pfTexCoordsCone[kt++] = 0.5;
		
		for (i = 0; i <= iSlices; i++) 
		{
			u = (GLfloat)(2 * PI - i * du);
			GLfloat c = cos(u);
			GLfloat s = sin(u);

			pfVerticesCone[kv] = c * fRadius;
			pfNormalsCone[kv++] = 0;

			pfVerticesCone[kv] = s * fRadius;
			pfNormalsCone[kv++] = 0;

			pfVerticesCone[kv] = -fHeight / 2;
			pfNormalsCone[kv++] = -1;
			
			pfTexCoordsCone[kt++] = (GLfloat)(0.5 - 0.5 * c);
			pfTexCoordsCone[kt++] = (GLfloat)(0.5 + 0.5 * s);
		}

		for (i = 0; i < iSlices; i++) 
		{
			piIndicesCone[k++] = startIndex;
			piIndicesCone[k++] = startIndex + i + 1;
			piIndicesCone[k++] = startIndex + i + 2;
		}
	}

	/*fprintf(gpFile, "Vertices\n");
	for (int i = 0; i < iVertexCount; i++)
	{
		fprintf(gpFile, "%f\t", pfVertices[i]);
	}

	fprintf(gpFile, "Indices\n");
	for (int i = 0; i < iTriangleCount * 3; i++)
	{
		fprintf(gpFile, "%d\t", piIndices[i]);
	}*/

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	glGenBuffers(1, &VBOPosition);
	glBindBuffer(GL_ARRAY_BUFFER, VBOPosition);
	glBufferData(GL_ARRAY_BUFFER, 3 * iVertexCountCone * sizeof(GLfloat *), pfVerticesCone, GL_STATIC_DRAW);
	glVertexAttribPointer(GL_ATTRIBUTE_VERTEX, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(GL_ATTRIBUTE_VERTEX);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &VBONormal);
	glBindBuffer(GL_ARRAY_BUFFER, VBONormal);
	glBufferData(GL_ARRAY_BUFFER, 3 * iVertexCountCone * sizeof(GLfloat *), pfNormalsCone, GL_STATIC_DRAW);
	glVertexAttribPointer(GL_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(GL_ATTRIBUTE_NORMAL);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &VBOTexture);
	glBindBuffer(GL_ARRAY_BUFFER, VBOTexture);
	glBufferData(GL_ARRAY_BUFFER, 2 * iVertexCountCone * sizeof(GLfloat *), pfTexCoordsCone, GL_STATIC_DRAW);
	glVertexAttribPointer(GL_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(GL_ATTRIBUTE_TEXTURE0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &VBOIndex);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBOIndex);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, iTriangleCountCone * 3 * sizeof(GLuint *), piIndicesCone, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	//return
	iIndicesCount = iTriangleCountCone * 3;

	free(pfVerticesCone);
	free(pfNormalsCone);
	free(pfTexCoordsCone);
	free(piIndicesCone);
}

void CCone::DrawCone(void)
{
	//bind Vao
	glBindVertexArray(VAO);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBOIndex);
	glDrawElements(GL_TRIANGLES, iIndicesCount, GL_UNSIGNED_INT, 0);

	//unbind Vao
	glBindVertexArray(0);
}

