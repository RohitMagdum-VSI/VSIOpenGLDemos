#pragma once

extern "C"
{
	typedef struct _SPHERE_DATA
	{
		FLOAT fRadius;
		UINT uiSlices;
		UINT uiStacks;

		UINT uiVerticesCount;

		FLOAT *pfVerticesSphere;
		FLOAT *pfNormalsSphere;
		FLOAT *pfTexCoordsSphere;

		UINT *puiIndicesSphere;

		UINT uiIndicesCount;

	}SPHERE_DATA, *P_SPHERE_DATA;

	typedef struct _RING_DATA
	{
		FLOAT *pfVerticesRing;
		FLOAT *pfNormalsRing;
		FLOAT *pfTexCoordsRing;
		UINT *puiIndicesRing;

		UINT uiVerticesCount;
		UINT uiIndicesCount;

	}RING_DATA, *P_RING_DATA;

	__declspec(dllexport) BOOL GetSphereData(FLOAT fRadius, UINT uiSlices, UINT uiStacks, SPHERE_DATA &SphereData, BOOL bInvertedNormals);
	__declspec(dllexport) void CleanupSphereData(SPHERE_DATA &Sphere);

	__declspec(dllexport) BOOL GetRingData(FLOAT fInnerRadius, FLOAT fOuterRadius, UINT uiSlices, RING_DATA &RingData);
	__declspec(dllexport) void CleanupRingData(RING_DATA &Ring);

}
