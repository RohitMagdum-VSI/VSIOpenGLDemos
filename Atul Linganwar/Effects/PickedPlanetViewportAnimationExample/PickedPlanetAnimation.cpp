#include <Windows.h>
#include <stdio.h>
#include <gl/glew.h>
#include <gl/GL.h>

#include "../common/vmath.h"
#include "../common/Common.h"
#include "Header.h"
#include "../common/Shapes.h"
#include "ObjectTransformations.h"
#include "Objects.h"
#include "EllipseData.h"
#include "BasicPlanetShader.h"
#include "PickedPlanetViewportAnimation.h"
#include "PickedPlanetAnimation.h"

extern PLANETS Planet;
extern bool gbIsAnimationDone;
extern bool gbIsRightMouseButtonPressed;
extern GLfloat gfEarthAngle;
extern vmath::mat4 gmat4ViewMatrix;
extern PLANET_AND_MOONS gpPlanetPick;
extern BASIC_PLANET_SHADER BasicPlanetShader;
extern vmath::mat4 gPerspectiveProjectionMatrix;
extern GLuint guiTexturePlanets[TOTAL_PLANETS];


void DrawPlanetPickAnimation(PLANET_AND_MOONS pick)
{
	switch (pick)
	{
	case SUN:
		AnimateSun();
		break;
	case MERCURY:
		AnimateMercury();
		break;
	case VENUS:
		AnimateVenus();
		break;
	case EARTH:
		AnimateEarth();
		break;
	case MARS:
		AnimateMars();
		break;
	case JUPITER:
		AnimateJupiter();
		break;
	case SATURN:
		AnimateSaturn();
		break;
	case URANUS:
		AnimateUranus();
		break;
	case NEPTUNE:
		AnimateNeptune();
		break;
	case PLUTO:
		AnimatePluto();
		break;
	default:
		break;
	}
}

void AnimateSun()
{
	static GLfloat fTranslateZ = 0.0f;

	glUseProgram(BasicPlanetShader.ShaderObject.uiShaderProgramObject);

	vmath::mat4 ModelMatrix = vmath::mat4::identity();
	vmath::mat4 ViewMatrix = vmath::mat4::identity();
	vmath::mat4 TranslationMatrix = vmath::mat4::identity();
	vmath::mat4 RotationMatrix = vmath::mat4::identity();
	vmath::mat4 ScaleMatrix = vmath::mat4::identity();

	ViewMatrix = vmath::lookat(vmath::vec3(0.0f, 0.0f, 170.0f), vmath::vec3(0.0f, 0.0f, 0.0f), vmath::vec3(0.0f, 1.0f, 0.0f));

	TranslationMatrix = vmath::translate(0.0f, 0.0f, fTranslateZ);
	ModelMatrix = ModelMatrix * TranslationMatrix;

	RotationMatrix = vmath::rotate(-90.0f, 1.0f, 0.0f, 0.0f); // after this y becomes -z and z becomes y
	RotationMatrix = RotationMatrix * vmath::rotate(gfEarthAngle, 0.0f, 0.0f, 1.0f);
	ModelMatrix = ModelMatrix * RotationMatrix;

	ScaleMatrix = vmath::scale(SCALE_FACTOR_SUN);
	ModelMatrix = ModelMatrix * ScaleMatrix;

	glUniformMatrix4fv(BasicPlanetShader.uiModelMatrixUniform, 1, GL_FALSE, ModelMatrix);
	glUniformMatrix4fv(BasicPlanetShader.uiViewMatrixUniform, 1, GL_FALSE, ViewMatrix);
	glUniformMatrix4fv(BasicPlanetShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, guiTexturePlanets[PLANET_AND_MOONS::SUN]);
	glUniform1i(BasicPlanetShader.uiTextureSamplerUniform, 0);

	DrawPlanet(&Planet);

	glUseProgram(0);

	static float exp = 0.0f;

	if (fTranslateZ < 120.0f)
	{
		fTranslateZ = fTranslateZ + exp;
		exp = exp + 0.001f;
	}

	if (gbIsRightMouseButtonPressed)
	{
		fTranslateZ = 0.0f;
		exp = 0.0f;
		gbIsAnimationDone = true;
		gbIsRightMouseButtonPressed = false;
	}
	// set gbIsAnimationDone after the fadeout not after button press.
}

void AnimateMercury()
{

}

void AnimateVenus()
{

}

void AnimateEarth()
{

}

void AnimateMars()
{

}

void AnimateJupiter()
{

}

void AnimateSaturn()
{

}

void AnimateUranus()
{

}

void AnimateNeptune()
{

}

void AnimatePluto()
{

}