#pragma once

#define SATURN_RING_INNER_RADIUS 800.0f
#define SATURN_RING_OUTER_RADIUS 1500.0f

typedef struct _SATURN_RING
{
	RING Ring;
}SATURN_RING, *P_SATURN_RING;

BOOL InitSaturnRing(SATURN_RING &SaturnRing, FLOAT fInnerRadius, FLOAT fOuterRadius, BOOL InvertedNormals = FALSE);
void DrawSaturnRing(SATURN_RING &SaturnRing, PLANET_SHADER &RingShader, vmath::mat4 modelMatrix, UINT ringTexture);
void CleanupSaturnRing(SATURN_RING &SaturnRing);
void DrawTransformedRing(FLOAT xPos, FLOAT yPos, FLOAT zPos, FLOAT fPlanetScale, GLuint Texture, BOOL bRotate);
