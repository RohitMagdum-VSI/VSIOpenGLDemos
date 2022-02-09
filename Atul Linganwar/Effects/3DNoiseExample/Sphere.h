#pragma once

typedef struct _SPHERE
{
	SPHERE_DATA SphereData;

	GLuint uiVAO;
	GLuint uiVBOPosition;
	GLuint uiVBOTexture;
	GLuint uiVBONormal;
	GLuint uiVBOElements;

}SPHERE;

bool InitializeSphere(GLfloat fRadius, GLint iSlices, GLint iStacks, SPHERE* pSphere);
void DrawSphere(SPHERE* pSphere);
void FreeSphere(SPHERE* pSphere);
