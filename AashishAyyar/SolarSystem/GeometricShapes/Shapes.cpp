#include <windows.h>
#include <stdio.h>
#include <math.h>

#include <gl/glew.h>
#include <gl/GL.h>

#include "vmath.h"
#include "Common.h"

#include "Shapes.h"

extern FILE *gpFile;

__declspec(dllexport) BOOL GetSphereData(FLOAT fRadius, UINT uiSlices, UINT uiStacks, SPHERE_DATA &SphereData, BOOL bInvertedNormals)
{
	FLOAT u, v, x, y, z;

	UINT IndexV = 0;
	UINT IndexT = 0;

	UINT k = 0;

	SphereData.uiVerticesCount = (uiSlices + 1) * (uiStacks + 1);
	SphereData.fRadius = fRadius;
	SphereData.uiSlices = uiSlices;
	SphereData.uiStacks = uiStacks;

	//
	//	Divide increment value for slices and stacks 
	//	2 * PI is circumference hence divide it with slices and stacks
	//
	
	FLOAT fSlicesIncrement = ((float)(2.0f * PI) / (float)uiSlices);
	FLOAT fStacksIncrement = ((float)PI / (float)uiStacks);

	SphereData.pfVerticesSphere = (FLOAT *)calloc(3 * SphereData.uiVerticesCount, sizeof(FLOAT));
	if (!SphereData.pfVerticesSphere)
	{
		CleanupSphereData(SphereData);
		return FALSE;
	}

	SphereData.pfNormalsSphere = (FLOAT *)calloc(3 * SphereData.uiVerticesCount, sizeof(FLOAT));
	if (!SphereData.pfNormalsSphere)
	{
		CleanupSphereData(SphereData);
		return FALSE;
	}

	SphereData.pfTexCoordsSphere = (FLOAT *)calloc(2 * SphereData.uiVerticesCount, sizeof(FLOAT));
	if (!SphereData.pfTexCoordsSphere)
	{
		CleanupSphereData(SphereData);
		return FALSE;
	}

	SphereData.puiIndicesSphere = (UINT *)calloc(2 * uiSlices * uiStacks * 3, sizeof(UINT));
	if (!SphereData.puiIndicesSphere)
	{
		CleanupSphereData(SphereData);
		return FALSE;
	}

	for (UINT i = 0; i <= uiStacks; i++) 
	{
		v = (FLOAT)(-PI / 2 + i * fStacksIncrement);
		for (UINT j = 0; j <= uiSlices; j++) 
		{
			u = j * fSlicesIncrement;
			x = cos(u) * cos(v);
			y = sin(u) * cos(v);
			z = sin(v);

			SphereData.pfVerticesSphere[IndexV] = fRadius * x;
			SphereData.pfNormalsSphere[IndexV++] = bInvertedNormals ? -x : x;

			SphereData.pfVerticesSphere[IndexV] = fRadius * y;
			SphereData.pfNormalsSphere[IndexV++] = bInvertedNormals ? -y : y;

			SphereData.pfVerticesSphere[IndexV] = fRadius * z;
			SphereData.pfNormalsSphere[IndexV++] = bInvertedNormals ? -z : z;

			SphereData.pfTexCoordsSphere[IndexT++] = (FLOAT)j / (FLOAT)uiSlices;
			SphereData.pfTexCoordsSphere[IndexT++] = (FLOAT)i / (FLOAT)uiStacks;
		}
	}

	for(UINT j = 0; j < uiStacks; j++)
	{
		UINT Row1 = j * (uiSlices + 1);		
		UINT Row2 = (j + 1) * (uiSlices + 1);

		for (UINT i = 0; i < uiSlices; i++)
		{
			//
			//	OGL Clockwise triangle
			//

			SphereData.puiIndicesSphere[k++] = Row1 + i;
			SphereData.puiIndicesSphere[k++] = Row2 + i + 1;
			SphereData.puiIndicesSphere[k++] = Row2 + i;

			SphereData.puiIndicesSphere[k++] = Row1 + i;
			SphereData.puiIndicesSphere[k++] = Row1 + i + 1;
			SphereData.puiIndicesSphere[k++] = Row2 + i + 1;
		}
	}

	SphereData.uiIndicesCount = 2 * uiSlices * uiStacks * 3;

	return TRUE;
}

__declspec(dllexport) void CleanupSphereData(SPHERE_DATA &Sphere)
{
	if (Sphere.pfVerticesSphere)
	{
		free(Sphere.pfVerticesSphere);
		Sphere.pfVerticesSphere = NULL;
	}

	if (Sphere.pfNormalsSphere)
	{
		free(Sphere.pfNormalsSphere);
		Sphere.pfNormalsSphere = NULL;
	}

	if (Sphere.pfTexCoordsSphere)
	{
		free(Sphere.pfTexCoordsSphere);
		Sphere.pfTexCoordsSphere = NULL;
	}

	if (Sphere.puiIndicesSphere)
	{
		free(Sphere.puiIndicesSphere);
		Sphere.puiIndicesSphere = NULL;
	}

	return;
}

//
//	Ellipse
//
__declspec(dllexport) BOOL GetEllipseData(UINT uiVerticesCount, UINT uiRadiusSemiMajorAxis, UINT uiRadiusSemiMinorAxis, ELLIPSE_DATA &EllipseData) 
{
	UINT IndexV = 0;
	UINT IndexC = 0;
	FLOAT fAngle = 0.0f;

	EllipseData.pfVerticesEllipse = (FLOAT *)calloc(3 * (uiVerticesCount + 2), sizeof(FLOAT));
	if (!EllipseData.pfVerticesEllipse)
	{
		CleanupEllipseData(EllipseData);
		return FALSE;
	}

	EllipseData.pfColorsEllipse = (FLOAT *)calloc(3 * (uiVerticesCount + 2), sizeof(FLOAT));
	if (!EllipseData.pfColorsEllipse)
	{
		CleanupEllipseData(EllipseData);
		return FALSE;
	}

	EllipseData.uiVerticesCount = uiVerticesCount + 1;

	for (int i = 0; i < uiVerticesCount + 2; i++) 
	{
		EllipseData.pfVerticesEllipse[IndexV++] = uiRadiusSemiMajorAxis * (FLOAT)cos(fAngle);
		EllipseData.pfVerticesEllipse[IndexV++] = 0;
		EllipseData.pfVerticesEllipse[IndexV++] = uiRadiusSemiMinorAxis * (FLOAT)sin(fAngle);
		
		if (i < uiVerticesCount + 2)
			fAngle = (2.0f * PI) * ((FLOAT)i / ((FLOAT)uiVerticesCount + 2));
		else
			fAngle = 0.0f;	

		EllipseData.pfColorsEllipse[IndexC++] = 0.1f;
		EllipseData.pfColorsEllipse[IndexC++] = 0.1f;
		EllipseData.pfColorsEllipse[IndexC++] = 0.1f;
	}
	
	return TRUE;
}

__declspec(dllexport) void CleanupEllipseData(ELLIPSE_DATA &EllipseData) 
{
	if (EllipseData.pfVerticesEllipse) 
	{
		free(EllipseData.pfVerticesEllipse);
		EllipseData.pfVerticesEllipse = NULL;
	}

	if (EllipseData.pfColorsEllipse)
	{
		free(EllipseData.pfColorsEllipse);
		EllipseData.pfColorsEllipse = NULL;
	}

	return;
}

//
//	Ring	
//
__declspec(dllexport) BOOL GetRingData(FLOAT fInnerRadius, FLOAT fOuterRadius, UINT uiSlices, RING_DATA &RingData)
{
	UINT IndexC = 0;
	FLOAT fAngle = 0.0f;
	FLOAT c, s;
	FLOAT d = 2 * PI / uiSlices;
	UINT IndexV = 0, IndexT = 0, IndexN = 0;
	int i;

	RingData.uiVerticesCount = (fInnerRadius == 0) ? uiSlices + 1 : uiSlices * 2;

	RingData.pfVerticesRing = (FLOAT *)calloc(3 * (RingData.uiVerticesCount), sizeof(FLOAT));
	if (!RingData.pfVerticesRing)
	{
		CleanupRingData(RingData);
		return FALSE;
	}

	RingData.pfNormalsRing = (FLOAT *)calloc(3 * (RingData.uiVerticesCount), sizeof(FLOAT));
	if (!RingData.pfNormalsRing)
	{
		CleanupRingData(RingData);
		return FALSE;
	}

	RingData.pfTexCoordsRing = (FLOAT *)calloc(2 * (RingData.uiVerticesCount), sizeof(FLOAT));
	if (!RingData.pfTexCoordsRing)
	{
		CleanupRingData(RingData);
		return FALSE;
	}

	RingData.uiIndicesCount = (fInnerRadius == 0) ? (3 * uiSlices) : 3 * 2 * uiSlices;

	RingData.puiIndicesRing = (UINT *)calloc(RingData.uiIndicesCount, sizeof(UINT));
	if (!RingData.puiIndicesRing)
	{
		CleanupRingData(RingData);
		return FALSE;
	}

	if (fInnerRadius == 0) 
	{
		for (i = 0; i < uiSlices; i++) 
		{
			c = (FLOAT)cos(d * i);
			s = (FLOAT)sin(d * i);

			RingData.pfVerticesRing[IndexV++] = c * fOuterRadius;
			RingData.pfVerticesRing[IndexV++] = s * fOuterRadius;
			RingData.pfVerticesRing[IndexV++] = 0;

			RingData.pfTexCoordsRing[IndexT++] = (FLOAT)(0.5 + 0.5 * c);
			RingData.pfTexCoordsRing[IndexT++] = (FLOAT)(0.5 + 0.5 * s);

			RingData.puiIndicesRing[IndexN++] = uiSlices;
			RingData.puiIndicesRing[IndexN++] = i;
			RingData.puiIndicesRing[IndexN++] = i == uiSlices - 1 ? 0 : i + 1;
		}
		
		RingData.pfVerticesRing[IndexV++] = RingData.pfVerticesRing[IndexV++] = RingData.pfVerticesRing[IndexV++] = 0;
		RingData.pfTexCoordsRing[IndexT++] = RingData.pfTexCoordsRing[IndexT++] = 0;
	}
	else
	{
		FLOAT r = fInnerRadius / fOuterRadius;

		for (i = 0; i < uiSlices; i++)
		{
			c = (FLOAT)cos(d * i);
			s = (FLOAT)sin(d * i);

			RingData.pfVerticesRing[IndexV++] = c * fInnerRadius;
			RingData.pfVerticesRing[IndexV++] = s * fInnerRadius;
			RingData.pfVerticesRing[IndexV++] = 0;

			RingData.pfTexCoordsRing[IndexT++] = (FLOAT)(0.5 + 0.5 * c * r);
			RingData.pfTexCoordsRing[IndexT++] = (FLOAT)(0.5 + 0.5 * s * r);

			RingData.pfVerticesRing[IndexV++] = c * fOuterRadius;
			RingData.pfVerticesRing[IndexV++] = s * fOuterRadius;
			RingData.pfVerticesRing[IndexV++] = 0;

 			RingData.pfTexCoordsRing[IndexT++] = (FLOAT)(0.5 + 0.5 * c);
			RingData.pfTexCoordsRing[IndexT++] = (FLOAT)(0.5 + 0.5 * s);
		}

		for (i = 0; i < uiSlices - 1; i++) 
		{
			RingData.puiIndicesRing[IndexN++] = 2 * i;
			RingData.puiIndicesRing[IndexN++] = 2 * i + 1;
			RingData.puiIndicesRing[IndexN++] = 2 * i + 3;
			RingData.puiIndicesRing[IndexN++] = 2 * i;
			RingData.puiIndicesRing[IndexN++] = 2 * i + 3;
			RingData.puiIndicesRing[IndexN++] = 2 * i + 2;
		}
	
		RingData.puiIndicesRing[IndexN++] = 2 * i;
		RingData.puiIndicesRing[IndexN++] = 2 * i + 1;
		RingData.puiIndicesRing[IndexN++] = 1;
		RingData.puiIndicesRing[IndexN++] = 2 * i;
		RingData.puiIndicesRing[IndexN++] = 1;
		RingData.puiIndicesRing[IndexN++] = 0;
	}

	for (i = 0; i < RingData.uiVerticesCount; i++) 
	{
		RingData.pfNormalsRing[3 * i] = RingData.pfNormalsRing[3 * i + 1] = 0;
		RingData.pfNormalsRing[3 * i + 2] = 1;
	}

	return TRUE;
}

__declspec(dllexport) void CleanupRingData(RING_DATA &RingData) 
{
	if (RingData.pfNormalsRing) 
	{
		free(RingData.pfNormalsRing);
		RingData.pfNormalsRing = NULL;
	}

	if (RingData.pfTexCoordsRing)
	{
		free(RingData.pfTexCoordsRing);
		RingData.pfTexCoordsRing = NULL;
	}

	if (RingData.pfVerticesRing)
	{
		free(RingData.pfVerticesRing);
		RingData.pfVerticesRing = NULL;
	}

	if (RingData.puiIndicesRing)
	{
		free(RingData.puiIndicesRing);
		RingData.puiIndicesRing = NULL;
	}
}