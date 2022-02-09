#pragma once

#define SPHERE_MAP_SCALE				100.0f

typedef struct _SPHERE_MAP
{
	SPHERE_DATA SphereData;

	GLuint uiVAO;
	GLuint uiVBOPosition;
	GLuint uiVBOTexture;
	GLuint uiVBOElements;

	OBJECT_TRANSFORMATION ObjectTransformation;

}SPHERE_MAP;

bool InitializeSphereMapData(GLfloat fRadius, GLint iSlices, GLint iStacks, SPHERE_MAP* pSphereMap);
void FreeSphereMapData(SPHERE_MAP* pSphereMap);

void DrawSphereMap(BASIC_TEXTURE_SHADER ShaderProgObj, SPHERE_MAP* pSphereMap, GLuint uiTexture);
void DrawSphereMap3D(BASIC_3DTEXTURE_SHADER ShaderProgObj, SPHERE_MAP* pSphereMap, GLuint uiTexture, GLuint uiDepthTexture);
