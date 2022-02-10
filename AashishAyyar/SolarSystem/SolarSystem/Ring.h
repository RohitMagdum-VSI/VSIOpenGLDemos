#pragma once

typedef struct _RING
{
	RING_DATA RingData;

	GLuint vaoRing;
	GLuint vboRingPosition;
	GLuint vboRingNormal;
	GLuint vboRingTexture;
	GLuint vboRingElements;

}RING, *P_RING;

//
//	InitRing: Allocate and initialize sphere, with specific radius
//
BOOL InitRing(FLOAT fInnerRadius, FLOAT fOuterRadius, RING &Ring, BOOL InvertedNormals);

BOOL InitVertexArrayAndBuffers(RING &Ring);

BOOL DrawRing(RING &Ring);

void CleanupRing(RING &Ring);
