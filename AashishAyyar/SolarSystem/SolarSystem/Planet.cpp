#include "Common.h"

extern PLANET gPlanet;
extern ORBIT gOrbit;
extern PLANET_SHADER gPlanetShader;
extern COLOR_SHADER gColorShader;
extern GLuint gPlanetTextures[9];
extern mat4 gPerspectiveProjectionMatrix;
extern mat4 gViewMatrix;

extern FLOAT gfPlanetAngle;	// Declared in Orbit.cpp
extern SATURN_RING gSaturnRing;

static GLfloat gfAmbientLight[] = { 0.0f, 0.0f, 0.0f, 1.0f };
static GLfloat gfDiffuseLight[] = { 1.0f, 1.0f, 1.0f, 1.0f };
static GLfloat gfSpecularLight[] = { 0.03f, 0.03f, 0.03f, 1.0f };
static GLfloat gfLightPosition[] = { 0.0f, 0.0f, 0.0f, 1.0f };

static GLfloat gfAmbientMaterial[] = { 0.0f, 0.0f, 0.0f, 1.0f };
static GLfloat gfDiffuseMaterial[] = { 1.0f, 1.0f, 1.0f, 1.0f };
static GLfloat gfSpecularMaterial[] = { 1.0f, 1.0f, 1.0f, 1.0f };
static GLfloat gfMaterialShininess = 10.0f;

FLOAT gXRot = 0.0f;
FLOAT gYRot = 0.0f;
FLOAT gZRot = 0.0f;

BOOL InitPlanet(PLANET &Planet, FLOAT SphereRadius, BOOL InvertedNormals)
{	
	BOOL bRetVal = FALSE;
	bRetVal = InitSphere(SphereRadius, Planet.Sphere, InvertedNormals);
	if (!bRetVal) 
	{
		CleanupPlanet(Planet);
		return FALSE;
	}

	return TRUE;
}

void DrawPlanet(PLANET &Planet, PLANET_SHADER &PlanetShader, vmath::mat4 modelMatrix, UINT planetTexture)
{
	glUseProgram(PlanetShader.Shader.renderProgram);

		glUniformMatrix4fv(PlanetShader.modelMatrixUniform, 1, GL_FALSE, modelMatrix);
		glUniformMatrix4fv(PlanetShader.viewMatrixUniform, 1, GL_FALSE, gViewMatrix);
		glUniformMatrix4fv(PlanetShader.projectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

		glUniform3fv(PlanetShader.laUniform, 1, gfAmbientLight);
		glUniform3fv(PlanetShader.ldUniform, 1, gfDiffuseLight);
		glUniform3fv(PlanetShader.lsUniform, 1, gfSpecularLight);
		glUniform4fv(PlanetShader.lightPosUniform, 1, gfLightPosition);

		glUniform3fv(PlanetShader.kaUniform, 1, gfAmbientMaterial);
		glUniform3fv(PlanetShader.kdUniform, 1, gfDiffuseMaterial);
		glUniform3fv(PlanetShader.ksUniform, 1, gfSpecularMaterial);
		glUniform1f(PlanetShader.materialShininessUniform, gfMaterialShininess);

		glActiveTexture(GL_TEXTURE0);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, planetTexture);
			glUniform1i(PlanetShader.textureSamplerUniform, 0);
			DrawSphere(Planet.Sphere);
		glBindTexture(GL_TEXTURE_2D, 0);

	glUseProgram(0);
}

void CleanupPlanet(PLANET &Planet) 
{
	CleanupSphere(Planet.Sphere);
}

FLOAT GetPlanetScale(PLANETS_AND_SATELLITES PlanetEnum)
{
	switch (PlanetEnum)
	{
	case SPHERE_MAP:
		return SCALE_SPHEREMAP_RADIUS * 2.0f;
		break;
	case SUN:
		return SCALE_SUN_RADIUS * 2.0f;
		break;
	case MERCURY:
		return SCALE_MERCURY_RADIUS * 10.0f;
		break;
	case VENUS:
		return SCALE_VENUS_RADIUS * 10.0f;
		break;
	case EARTH:
		return SCALE_EARTH_RADIUS * 10.0f;
		break;
	case MARS:
		return SCALE_MARS_RADIUS * 10.0f;
		break;
	case JUPITER:
		return SCALE_JUPITER_RADIUS;
		break;
	case SATURN:
		return SCALE_SATURN_RADIUS;
		break;
	case URANUS:
		return SCALE_URANUS_RADIUS;
		break;
	case NEPTUNE:
		return SCALE_NEPTUNE_RADIUS;
		break;
	case PLUTO:
		return SCALE_PLUTO_RADIUS * 10.0f;
		break;
	default:
		return 1.0f;
		break;
	}
}

void DrawTransformedPlanet(FLOAT xPos, FLOAT yPos, FLOAT zPos, FLOAT fPlanetScale, GLuint Texture, BOOL bRotate)
{
	mat4 translationMatrix = mat4::identity();
	mat4 rotationMatrix = mat4::identity();
	mat4 scaleMatrix = mat4::identity();
	mat4 modelMatrix = mat4::identity();

	static float fAngle = 0.0f;
	translationMatrix = translate(xPos, yPos, zPos);
	rotationMatrix *= rotate(270.0f, 1.0f, 0.0f, 0.0f);
	
	if(bRotate)
		rotationMatrix *= rotate(fAngle, 0.0f, 0.0f, 1.0f);
	
	scaleMatrix = scale(fPlanetScale, fPlanetScale, fPlanetScale);

	modelMatrix = translationMatrix * rotationMatrix * scaleMatrix;
	DrawPlanet(gPlanet, gPlanetShader, modelMatrix, Texture);
	
	fAngle += 0.1;
}

void DrawAllPlanets()
{
	//
	//	Sun
	//
	DrawTransformedPlanet(
		0.0f,
		0.0f,
		0.0f, 
		GetPlanetScale(PLANETS_AND_SATELLITES::SUN), 
		gPlanetTextures[PLANETS_AND_SATELLITES::SUN],
		FALSE);

	//
	//	Mercury
	//
	DrawTransformedPlanet(
		GetPlanetXPosition(gOrbit, PLANETS_AND_SATELLITES::MERCURY, ANGLE_WITH_OFFSET(gfPlanetAngle, MERCURY_OFFSET)),
		0.0f,
		GetPlanetZPosition(gOrbit, PLANETS_AND_SATELLITES::MERCURY, ANGLE_WITH_OFFSET(gfPlanetAngle, MERCURY_OFFSET)),
		GetPlanetScale(PLANETS_AND_SATELLITES::MERCURY), 
		gPlanetTextures[PLANETS_AND_SATELLITES::MERCURY]
	);

	//
	//	Venus
	//
	DrawTransformedPlanet(
		GetPlanetXPosition(gOrbit, PLANETS_AND_SATELLITES::VENUS, ANGLE_WITH_OFFSET(gfPlanetAngle, VENUS_OFFSET)),
		0.0f,
		GetPlanetZPosition(gOrbit, PLANETS_AND_SATELLITES::VENUS, ANGLE_WITH_OFFSET(gfPlanetAngle, VENUS_OFFSET)),
		GetPlanetScale(PLANETS_AND_SATELLITES::VENUS),
		gPlanetTextures[PLANETS_AND_SATELLITES::VENUS]
	);

	//
	//	Earth
	//
	DrawTransformedPlanet(
		GetPlanetXPosition(gOrbit, PLANETS_AND_SATELLITES::EARTH, ANGLE_WITH_OFFSET(gfPlanetAngle, EARTH_OFFSET)),
		0.0f,
		GetPlanetZPosition(gOrbit, PLANETS_AND_SATELLITES::EARTH, ANGLE_WITH_OFFSET(gfPlanetAngle, EARTH_OFFSET)),
		GetPlanetScale(PLANETS_AND_SATELLITES::EARTH),
		gPlanetTextures[PLANETS_AND_SATELLITES::EARTH]
	);

	//
	//	Mars
	//
	DrawTransformedPlanet(
		GetPlanetXPosition(gOrbit, PLANETS_AND_SATELLITES::MARS, ANGLE_WITH_OFFSET(gfPlanetAngle, MARS_OFFSET)),
		0.0f,
		GetPlanetZPosition(gOrbit, PLANETS_AND_SATELLITES::MARS, ANGLE_WITH_OFFSET(gfPlanetAngle, MARS_OFFSET)),
		GetPlanetScale(PLANETS_AND_SATELLITES::MARS),
		gPlanetTextures[PLANETS_AND_SATELLITES::MARS]
	);

	//
	//	Jupiter
	//
	DrawTransformedPlanet(
		GetPlanetXPosition(gOrbit, PLANETS_AND_SATELLITES::JUPITER, ANGLE_WITH_OFFSET(gfPlanetAngle, JUPITER_OFFSET)),
		0.0f,
		GetPlanetZPosition(gOrbit, PLANETS_AND_SATELLITES::JUPITER, ANGLE_WITH_OFFSET(gfPlanetAngle, JUPITER_OFFSET)),
		GetPlanetScale(PLANETS_AND_SATELLITES::JUPITER),
		gPlanetTextures[PLANETS_AND_SATELLITES::JUPITER]
	);

	//
	//	Saturn
	//


	DrawTransformedRing(
		GetPlanetXPosition(gOrbit, PLANETS_AND_SATELLITES::SATURN, ANGLE_WITH_OFFSET(gfPlanetAngle, SATURN_OFFSET)),
		0000.0f,
		GetPlanetZPosition(gOrbit, PLANETS_AND_SATELLITES::SATURN, ANGLE_WITH_OFFSET(gfPlanetAngle, SATURN_OFFSET)),
		1.0f,
		gPlanetTextures[PLANETS_AND_SATELLITES::SATURN_RING_ID],
		TRUE
	);

	DrawTransformedPlanet(
		GetPlanetXPosition(gOrbit, PLANETS_AND_SATELLITES::SATURN, ANGLE_WITH_OFFSET(gfPlanetAngle, SATURN_OFFSET)),
		0.0f,
		GetPlanetZPosition(gOrbit, PLANETS_AND_SATELLITES::SATURN, ANGLE_WITH_OFFSET(gfPlanetAngle, SATURN_OFFSET)),
		GetPlanetScale(PLANETS_AND_SATELLITES::SATURN),
		gPlanetTextures[PLANETS_AND_SATELLITES::SATURN]
	);

	//
	//	Uranus
	//
	DrawTransformedPlanet(
		GetPlanetXPosition(gOrbit, PLANETS_AND_SATELLITES::URANUS, ANGLE_WITH_OFFSET(gfPlanetAngle, URANUS_OFFSET)),
		0.0f,
		GetPlanetZPosition(gOrbit, PLANETS_AND_SATELLITES::URANUS, ANGLE_WITH_OFFSET(gfPlanetAngle, URANUS_OFFSET)),
		GetPlanetScale(PLANETS_AND_SATELLITES::URANUS),
		gPlanetTextures[PLANETS_AND_SATELLITES::URANUS]
	);

	//
	//	Neptune
	//
	DrawTransformedPlanet(
		GetPlanetXPosition(gOrbit, PLANETS_AND_SATELLITES::NEPTUNE, ANGLE_WITH_OFFSET(gfPlanetAngle, NEPTUNE_OFFSET)),
		0.0f,
		GetPlanetZPosition(gOrbit, PLANETS_AND_SATELLITES::NEPTUNE, ANGLE_WITH_OFFSET(gfPlanetAngle, NEPTUNE_OFFSET)),
		GetPlanetScale(PLANETS_AND_SATELLITES::NEPTUNE),
		gPlanetTextures[PLANETS_AND_SATELLITES::NEPTUNE]
	);

	//
	//	Pluto
	//
	DrawTransformedPlanet(
		GetPlanetXPosition(gOrbit, PLANETS_AND_SATELLITES::PLUTO, ANGLE_WITH_OFFSET(gfPlanetAngle, PLUTO_OFFSET)),
		0.0f,
		GetPlanetZPosition(gOrbit, PLANETS_AND_SATELLITES::PLUTO, ANGLE_WITH_OFFSET(gfPlanetAngle, PLUTO_OFFSET)),
		GetPlanetScale(PLANETS_AND_SATELLITES::PLUTO),
		gPlanetTextures[PLANETS_AND_SATELLITES::PLUTO]
	);

	return ;
}

FLOAT GetPlanetOffset(PLANETS_AND_SATELLITES Planet)
{
	switch (Planet)
	{
	case MERCURY:
		return MERCURY_OFFSET;
		break;
	case VENUS:
		return VENUS_OFFSET;
		break;
	case EARTH:
		return EARTH_OFFSET;
		break;
	case MARS:
		return MARS_OFFSET;
		break;
	case JUPITER:
		return JUPITER_OFFSET;
		break;
	case SATURN:
		return SATURN_OFFSET;
		break;
	case URANUS:
		return URANUS_OFFSET;
		break;
	case NEPTUNE:
		return NEPTUNE_OFFSET;
		break;
	case PLUTO:
		return PLUTO_OFFSET;
		break;
	default:
		return 0;
		break;
	}

	return 0;
}
