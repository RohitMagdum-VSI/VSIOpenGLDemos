#include "Common.h"

extern mat4 gPerspectiveProjectionMatrix;
extern mat4 gViewMatrix;

extern FLOAT gfPlanetAngle;	// Declared in Orbit.cpp

extern SATURN_RING gSaturnRing;
extern PLANET_SHADER gPlanetShader;

extern GLuint gPlanetTextures[9];

static GLfloat gfAmbientLight[] = { 0.0f, 0.0f, 0.0f, 1.0f };
static GLfloat gfDiffuseLight[] = { 1.0f, 1.0f, 1.0f, 1.0f };
static GLfloat gfSpecularLight[] = { 0.03f, 0.03f, 0.03f, 1.0f };
static GLfloat gfLightPosition[] = { 0.0f, 0.0f, 0.0f, 1.0f };

static GLfloat gfAmbientMaterial[] = { 0.0f, 0.0f, 0.0f, 1.0f };
static GLfloat gfDiffuseMaterial[] = { 1.0f, 1.0f, 1.0f, 1.0f };
static GLfloat gfSpecularMaterial[] = { 1.0f, 1.0f, 1.0f, 1.0f };
static GLfloat gfMaterialShininess = 10.0f;

BOOL InitSaturnRing(SATURN_RING &SaturnRing, FLOAT fInnerRadius, FLOAT fOuterRadius, BOOL InvertedNormals)
{
	BOOL bRetVal = FALSE;
	bRetVal = InitRing(fInnerRadius, fOuterRadius, SaturnRing.Ring, InvertedNormals);
	if (!bRetVal)
	{
		CleanupSaturnRing(SaturnRing);
		return FALSE;
	}

	return TRUE;
}

void DrawSaturnRing(SATURN_RING &SaturnRing, PLANET_SHADER &RingShader, vmath::mat4 modelMatrix, UINT RingTexture) 
{
	glUseProgram(RingShader.Shader.renderProgram);

		glUniformMatrix4fv(RingShader.modelMatrixUniform, 1, GL_FALSE, modelMatrix);
		glUniformMatrix4fv(RingShader.viewMatrixUniform, 1, GL_FALSE, gViewMatrix);
		glUniformMatrix4fv(RingShader.projectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

		glUniform3fv(RingShader.laUniform, 1, gfAmbientLight);
		glUniform3fv(RingShader.ldUniform, 1, gfDiffuseLight);
		glUniform3fv(RingShader.lsUniform, 1, gfSpecularLight);
		glUniform4fv(RingShader.lightPosUniform, 1, gfLightPosition);

		glUniform3fv(RingShader.kaUniform, 1, gfAmbientMaterial);
		glUniform3fv(RingShader.kdUniform, 1, gfDiffuseMaterial);
		glUniform3fv(RingShader.ksUniform, 1, gfSpecularMaterial);
		glUniform1f(RingShader.materialShininessUniform, gfMaterialShininess);

		glActiveTexture(GL_TEXTURE0);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, RingTexture);
			glUniform1i(RingShader.textureSamplerUniform, 0);
			DrawRing(SaturnRing.Ring);
		glBindTexture(GL_TEXTURE_2D, 0);

	glUseProgram(0);
}

void DrawTransformedRing(FLOAT xPos, FLOAT yPos, FLOAT zPos, FLOAT fPlanetScale, GLuint Texture, BOOL bRotate)
{
	mat4 translationMatrix = mat4::identity();
	mat4 rotationMatrix = mat4::identity();
	mat4 scaleMatrix = mat4::identity();
	mat4 modelMatrix = mat4::identity();

	static float fAngle = 0.0f;
	translationMatrix = translate(xPos, yPos, zPos);
	rotationMatrix *= rotate(270.0f, 1.0f, 0.0f, 0.0f);

	if (bRotate)
		rotationMatrix *= rotate(fAngle, 0.0f, 0.0f, 1.0f);

	scaleMatrix = scale(fPlanetScale, fPlanetScale, fPlanetScale);

	modelMatrix = translationMatrix * rotationMatrix * scaleMatrix;
	DrawSaturnRing(gSaturnRing, gPlanetShader, modelMatrix, gPlanetTextures[PLANETS_AND_SATELLITES::SATURN_RING_ID]);

	fAngle += 0.1;
}

void CleanupSaturnRing(SATURN_RING &SaturnRing) 
{
	CleanupRing(SaturnRing.Ring);
}
