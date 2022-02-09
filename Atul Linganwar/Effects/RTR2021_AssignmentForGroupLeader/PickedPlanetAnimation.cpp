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
#include "BasicColorShader.h"
#include "BasicPlanetShader.h"
#include "BasicQuadRTTShader.h"
#include "FontsMap.h"
#include "SolarSystem.h"
#include "FrameBuffers.h"
#include "PickedPlanetAnimation.h"
#include "SunShader.h"
#include "MarbalShader.h"

#define ANIMATION_CAMERA_X			0.0f
#define ANIMATION_CAMERA_Y			60.0f
#define ANIMATION_CAMERA_Z			160.0f

#define TRANSLATION_EXP_INCREATE	0.01f

extern PLANETS Planet;
extern SATURN_RING SaturnRing;
extern PLANETS SphereMoon;
extern GLfloat gfAngle;
extern GLfloat gfMoonAngle;
extern FONT_MAP FontMapArial;
extern GLfloat gWindowWidth;
extern GLfloat gWindowHeight;
extern bool gbIsAnimationDone;
extern FONT_SHADER FontShader;
extern vmath::mat4 gmat4ViewMatrix;
extern PLANET_AND_MOONS gpPlanetPick;
extern bool gbIsRightMouseButtonPressed;
extern BASIC_COLOR_SHADER BasicColorShader;
extern BASIC_PLANET_SHADER BasicPlanetShader;
extern GLuint guiTexturePlanets[TOTAL_PLANETS];
extern vmath::mat4 gPerspectiveProjectionMatrix;
extern vmath::mat4 gOrthographicProjectionMatrix;
extern BASIC_QUAD_RTT_TEXTURE_SHADER BasicQuadRTTTextureShader;

extern GLfloat gfAmbientLight[];
extern GLfloat gfDiffuseLight[];
extern GLfloat gfSpecularLight[];
extern GLfloat gfLightPosition[];

extern GLfloat gfAmbientMaterial[];
extern GLfloat gfDiffuseMaterial[];
extern GLfloat gfSpecularMaterial[];
extern GLfloat gfMaterialShininess;

extern GLuint guiVAO;
extern GLuint guiVBOPos;
extern GLuint guiVBOColor;

extern GLfloat gfSunRadius;
extern GLfloat gfMoonRadius;

extern GLuint guiVAORTT;
extern GLuint guiVBOTextureRTT;
extern GLuint guiVBOPositionRTT;

extern SUN_SHADER SunShader;
extern MARBAL_SHADER MarbalShader;

extern FRAMEBUFFER_OBJECT gFBOViewport;

extern GLuint noise3DTexName;

GLfloat gfMoonTranslationX = 0.0f;
GLfloat gfMoonTranslationZ = 0.0f;
bool gbIncrease = true;
float fOffset = 0.791508973f;

PICKING_DATA PickingData;
bool gbIsFadeOutDone = false;
bool gbIsPickTransitionDone = false;

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
	static float exp = 0.0f;
	static GLfloat fTranslateZ = 0.0f;

	POINT_TRACE p1 = { 0.0f, 0.0f, 0.0f };
	POINT_TRACE p2 = { ANIMATION_CAMERA_X, ANIMATION_CAMERA_Y, ANIMATION_CAMERA_Z - 10.0f };

	POINT_TRACE p = GetPointOnLine(p1, p2, fTranslateZ);

	if (false == gbIsPickTransitionDone)
	{
		if (p.z < ANIMATION_CAMERA_Z - 10.0f)
		{
			fTranslateZ = fTranslateZ + exp;
			exp = exp + TRANSLATION_EXP_INCREATE;
		}
		else
		{
			fTranslateZ = 0.0f;
			exp = 0.0f;
			gbIsPickTransitionDone = true;
		}

		glUseProgram(BasicPlanetShader.ShaderObject.uiShaderProgramObject);

		vmath::mat4 ModelMatrix = vmath::mat4::identity();
		vmath::mat4 ViewMatrix = vmath::mat4::identity();
		vmath::mat4 TranslationMatrix = vmath::mat4::identity();
		vmath::mat4 RotationMatrix = vmath::mat4::identity();
		vmath::mat4 ScaleMatrix = vmath::mat4::identity();

		TranslationMatrix = vmath::translate(p.x, p.y, p.z);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		RotationMatrix = vmath::rotate(-90.0f, 1.0f, 0.0f, 0.0f); // after this y becomes -z and z becomes y
		RotationMatrix = RotationMatrix * vmath::rotate(gfAngle, 0.0f, 0.0f, 1.0f);
		ModelMatrix = ModelMatrix * RotationMatrix;

		ScaleMatrix = vmath::scale(SCALE_FACTOR_SUN);
		ModelMatrix = ModelMatrix * ScaleMatrix;

		glUniformMatrix4fv(BasicPlanetShader.uiModelMatrixUniform, 1, GL_FALSE, ModelMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiViewMatrixUniform, 1, GL_FALSE, gmat4ViewMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, guiTexturePlanets[PLANET_AND_MOONS::SUN]);
		glUniform1i(BasicPlanetShader.uiTextureSamplerUniform, 0);

		glUniform3fv(BasicPlanetShader.uiLaUniform, 1, gfAmbientLight);
		glUniform3fv(BasicPlanetShader.uiLdUniform, 1, gfDiffuseLight);
		glUniform3fv(BasicPlanetShader.uiLsUniform, 1, gfSpecularLight);
		glUniform4fv(BasicPlanetShader.uiLightPositionUniform, 1, gfLightPosition);

		glUniform3fv(BasicPlanetShader.uiKaUniform, 1, gfAmbientMaterial);
		glUniform3fv(BasicPlanetShader.uiKdUniform, 1, gfDiffuseMaterial);
		glUniform3fv(BasicPlanetShader.uiKsUniform, 1, gfSpecularMaterial);
		glUniform1f(BasicPlanetShader.uiMaterialShininessUniform, gfMaterialShininess);

		DrawPlanet(&Planet);

		glUseProgram(0);
	}
	else
	{
		DrawNoiseSun(1.0f);
		//RenderPlanetOntoABorderedViewport(guiTexturePlanets[PLANET_AND_MOONS::SUN], 1.0f, 0, 0, false);
		ShowSunViewport();
	}
}

void AnimateMercury()
{
	static float exp = 0.0f;
	static GLfloat fTranslateZ = 0.0f;

	GLfloat x = gpEllipsePathMercury->EllipseData.fSemiMajorAxis * (GLfloat)cos(PickingData.gfAngleTranslate);
	GLfloat z = -gpEllipsePathMercury->EllipseData.fSemiMinorAxis * (GLfloat)sin(PickingData.gfAngleTranslate);

	POINT_TRACE p1 = { x, 0.0f, z };
	POINT_TRACE p2 = { ANIMATION_CAMERA_X, ANIMATION_CAMERA_Y, ANIMATION_CAMERA_Z };

	POINT_TRACE p = GetPointOnLine(p1, p2, z + fTranslateZ);

	if (false == gbIsPickTransitionDone)
	{
		if (p.z < ANIMATION_CAMERA_Z)
		{
			fTranslateZ = fTranslateZ + exp;
			exp = exp + TRANSLATION_EXP_INCREATE;
		}
		else
		{
			fTranslateZ = 0.0f;
			exp = 0.0f;
			gbIsPickTransitionDone = true;
		}

		glUseProgram(BasicPlanetShader.ShaderObject.uiShaderProgramObject);

		vmath::mat4 ModelMatrix = vmath::mat4::identity();
		vmath::mat4 ViewMatrix = vmath::mat4::identity();
		vmath::mat4 TranslationMatrix = vmath::mat4::identity();
		vmath::mat4 RotationMatrix = vmath::mat4::identity();
		vmath::mat4 ScaleMatrix = vmath::mat4::identity();

		TranslationMatrix = vmath::translate(p.x, p.y, p.z);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		RotationMatrix = vmath::rotate(-90.0f, 1.0f, 0.0f, 0.0f); // after this y becomes -z and z becomes y
		RotationMatrix = RotationMatrix * vmath::rotate(gfAngle, 0.0f, 0.0f, 1.0f);
		ModelMatrix = ModelMatrix * RotationMatrix;

		ScaleMatrix = vmath::scale(SCALE_FACTOR_MERCURY);
		ModelMatrix = ModelMatrix * ScaleMatrix;

		glUniformMatrix4fv(BasicPlanetShader.uiModelMatrixUniform, 1, GL_FALSE, ModelMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiViewMatrixUniform, 1, GL_FALSE, gmat4ViewMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, guiTexturePlanets[PLANET_AND_MOONS::MERCURY]);
		glUniform1i(BasicPlanetShader.uiTextureSamplerUniform, 0);

		glUniform3fv(BasicPlanetShader.uiLaUniform, 1, gfAmbientLight);
		glUniform3fv(BasicPlanetShader.uiLdUniform, 1, gfDiffuseLight);
		glUniform3fv(BasicPlanetShader.uiLsUniform, 1, gfSpecularLight);
		glUniform4fv(BasicPlanetShader.uiLightPositionUniform, 1, gfLightPosition);

		glUniform3fv(BasicPlanetShader.uiKaUniform, 1, gfAmbientMaterial);
		glUniform3fv(BasicPlanetShader.uiKdUniform, 1, gfDiffuseMaterial);
		glUniform3fv(BasicPlanetShader.uiKsUniform, 1, gfSpecularMaterial);
		glUniform1f(BasicPlanetShader.uiMaterialShininessUniform, gfMaterialShininess);

		DrawPlanet(&Planet);

		glUseProgram(0);
	}
	else
	{
		DrawNoiseMarble(SCALE_FACTOR_MERCURY, 0.0f, false, vmath::vec3(0.0f, 0.0f, 0.0f), vmath::vec3(0.0f, 0.0f, 0.0f));
		//RenderPlanetOntoABorderedViewport(guiTexturePlanets[PLANET_AND_MOONS::MERCURY], SCALE_FACTOR_MERCURY, 0, 0.0f, true);
		ShowMercuryViewport();
	}
}

void AnimateVenus()
{
	static float exp = 0.0f;
	static GLfloat fTranslateZ = 0.0f;

	GLfloat x = gpEllipsePathVenus->EllipseData.fSemiMajorAxis * (GLfloat)cos(PickingData.gfAngleTranslate);
	GLfloat z = -gpEllipsePathVenus->EllipseData.fSemiMinorAxis * (GLfloat)sin(PickingData.gfAngleTranslate);

	POINT_TRACE p1 = { x, 0.0f, z };
	POINT_TRACE p2 = { ANIMATION_CAMERA_X, ANIMATION_CAMERA_Y, ANIMATION_CAMERA_Z };

	POINT_TRACE p = GetPointOnLine(p1, p2, z + fTranslateZ);

	if (false == gbIsPickTransitionDone)
	{
		if (p.z < ANIMATION_CAMERA_Z)
		{
			fTranslateZ = fTranslateZ + exp;
			exp = exp + TRANSLATION_EXP_INCREATE;
		}
		else
		{
			fTranslateZ = 0.0f;
			exp = 0.0f;
			gbIsPickTransitionDone = true;
		}

		glUseProgram(BasicPlanetShader.ShaderObject.uiShaderProgramObject);

		vmath::mat4 ModelMatrix = vmath::mat4::identity();
		vmath::mat4 ViewMatrix = vmath::mat4::identity();
		vmath::mat4 TranslationMatrix = vmath::mat4::identity();
		vmath::mat4 RotationMatrix = vmath::mat4::identity();
		vmath::mat4 ScaleMatrix = vmath::mat4::identity();

		TranslationMatrix = vmath::translate(p.x, p.y, p.z);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		RotationMatrix = vmath::rotate(-90.0f, 1.0f, 0.0f, 0.0f); // after this y becomes -z and z becomes y
		RotationMatrix = RotationMatrix * vmath::rotate(gfAngle, 0.0f, 0.0f, 1.0f);
		ModelMatrix = ModelMatrix * RotationMatrix;

		ScaleMatrix = vmath::scale(SCALE_FACTOR_VENUS);
		ModelMatrix = ModelMatrix * ScaleMatrix;

		glUniformMatrix4fv(BasicPlanetShader.uiModelMatrixUniform, 1, GL_FALSE, ModelMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiViewMatrixUniform, 1, GL_FALSE, gmat4ViewMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, guiTexturePlanets[PLANET_AND_MOONS::VENUS]);
		glUniform1i(BasicPlanetShader.uiTextureSamplerUniform, 0);

		glUniform3fv(BasicPlanetShader.uiLaUniform, 1, gfAmbientLight);
		glUniform3fv(BasicPlanetShader.uiLdUniform, 1, gfDiffuseLight);
		glUniform3fv(BasicPlanetShader.uiLsUniform, 1, gfSpecularLight);
		glUniform4fv(BasicPlanetShader.uiLightPositionUniform, 1, gfLightPosition);

		glUniform3fv(BasicPlanetShader.uiKaUniform, 1, gfAmbientMaterial);
		glUniform3fv(BasicPlanetShader.uiKdUniform, 1, gfDiffuseMaterial);
		glUniform3fv(BasicPlanetShader.uiKsUniform, 1, gfSpecularMaterial);
		glUniform1f(BasicPlanetShader.uiMaterialShininessUniform, gfMaterialShininess);

		DrawPlanet(&Planet);

		glUseProgram(0);
	}
	else
	{
		DrawNoiseMarble(SCALE_FACTOR_VENUS, 0.0f, false, vmath::vec3(0.0f, 0.0f, 0.0f), vmath::vec3(0.0f, 0.0f, 0.0f));
		//RenderPlanetOntoABorderedViewport(guiTexturePlanets[PLANET_AND_MOONS::VENUS], SCALE_FACTOR_VENUS, 0, 0.0f, true);
		ShowVenusViewport();
	}
}

void AnimateEarth()
{
	static float exp = 0.0f;
	static GLfloat fTranslateZ = 0.0f;

	GLfloat x = gpEllipsePathEarth->EllipseData.fSemiMajorAxis * (GLfloat)cos(PickingData.gfAngleTranslate);
	GLfloat z = -gpEllipsePathEarth->EllipseData.fSemiMinorAxis * (GLfloat)sin(PickingData.gfAngleTranslate);

	POINT_TRACE p1 = { x, 0.0f, z };
	POINT_TRACE p2 = { ANIMATION_CAMERA_X, ANIMATION_CAMERA_Y, ANIMATION_CAMERA_Z };

	POINT_TRACE p = GetPointOnLine(p1, p2, z + fTranslateZ);
	
	if (false == gbIsPickTransitionDone)
	{
		if (p.z < ANIMATION_CAMERA_Z)
		{
			fTranslateZ = fTranslateZ + exp;
			exp = exp + TRANSLATION_EXP_INCREATE;
		}
		else
		{
			fTranslateZ = 0.0f;
			exp = 0.0f;
			gbIsPickTransitionDone = true;
		}

		glUseProgram(BasicPlanetShader.ShaderObject.uiShaderProgramObject);

		vmath::mat4 ModelMatrix = vmath::mat4::identity();
		vmath::mat4 ViewMatrix = vmath::mat4::identity();
		vmath::mat4 TranslationMatrix = vmath::mat4::identity();
		vmath::mat4 RotationMatrix = vmath::mat4::identity();
		vmath::mat4 ScaleMatrix = vmath::mat4::identity();

		TranslationMatrix = vmath::translate(p.x, p.y, p.z);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		RotationMatrix = vmath::rotate(-90.0f, 1.0f, 0.0f, 0.0f); // after this y becomes -z and z becomes y
		RotationMatrix = RotationMatrix * vmath::rotate(gfAngle, 0.0f, 0.0f, 1.0f);
		ModelMatrix = ModelMatrix * RotationMatrix;

		ScaleMatrix = vmath::scale(SCALE_FACTOR_EARTH);
		ModelMatrix = ModelMatrix * ScaleMatrix;

		glUniformMatrix4fv(BasicPlanetShader.uiModelMatrixUniform, 1, GL_FALSE, ModelMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiViewMatrixUniform, 1, GL_FALSE, gmat4ViewMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, guiTexturePlanets[PLANET_AND_MOONS::EARTH]);
		glUniform1i(BasicPlanetShader.uiTextureSamplerUniform, 0);

		glUniform3fv(BasicPlanetShader.uiLaUniform, 1, gfAmbientLight);
		glUniform3fv(BasicPlanetShader.uiLdUniform, 1, gfDiffuseLight);
		glUniform3fv(BasicPlanetShader.uiLsUniform, 1, gfSpecularLight);
		glUniform4fv(BasicPlanetShader.uiLightPositionUniform, 1, gfLightPosition);

		glUniform3fv(BasicPlanetShader.uiKaUniform, 1, gfAmbientMaterial);
		glUniform3fv(BasicPlanetShader.uiKdUniform, 1, gfDiffuseMaterial);
		glUniform3fv(BasicPlanetShader.uiKsUniform, 1, gfSpecularMaterial);
		glUniform1f(BasicPlanetShader.uiMaterialShininessUniform, gfMaterialShininess);
		
		DrawPlanet(&Planet);

		//moon
		ModelMatrix = vmath::mat4::identity();
		TranslationMatrix = vmath::mat4::identity();
		RotationMatrix = vmath::mat4::identity();
		ScaleMatrix = vmath::mat4::identity();

		gfMoonTranslationX = (gfSunRadius * SCALE_FACTOR_EARTH + gfMoonRadius + 0.5f) * (GLfloat)cos(gfMoonAngleTranslate);
		gfMoonTranslationZ = (gfSunRadius * SCALE_FACTOR_EARTH + gfMoonRadius + 0.5f) * (GLfloat)sin(gfMoonAngleTranslate);

		TranslationMatrix = vmath::mat4::identity();
		TranslationMatrix = vmath::translate(p.x, p.y, p.z);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		TranslationMatrix = vmath::mat4::identity();
		TranslationMatrix = vmath::translate(gfMoonTranslationX, 0.0f, -gfMoonTranslationZ);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		RotationMatrix = vmath::rotate(-90.0f, 1.0f, 0.0f, 0.0f);
		RotationMatrix = RotationMatrix * vmath::rotate(gfMoonAngle, 0.0f, 0.0f, 1.0f);
		ModelMatrix = ModelMatrix * RotationMatrix;

		ScaleMatrix = vmath::scale(SCALE_FACTOR_EARTH_MOON);
		ModelMatrix = ModelMatrix * ScaleMatrix;

		glUniformMatrix4fv(BasicPlanetShader.uiModelMatrixUniform, 1, GL_FALSE, ModelMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiViewMatrixUniform, 1, GL_FALSE, gmat4ViewMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, guiTexturePlanets[PLANET_AND_MOONS::EARTH_MOON]);
		glUniform1i(BasicPlanetShader.uiTextureSamplerUniform, 0);
		
		DrawPlanet(&SphereMoon);

		glUseProgram(0);
	}
	else
	{
		DrawNoiseMarble(SCALE_FACTOR_EARTH, SCALE_FACTOR_EARTH_MOON, true, vmath::vec3(0.0f, 0.0f, 0.0f), vmath::vec3(0.0f, 0.0f, 0.0f));
		//RenderPlanetOntoABorderedViewport(guiTexturePlanets[PLANET_AND_MOONS::EARTH], SCALE_FACTOR_EARTH, guiTexturePlanets[PLANET_AND_MOONS::EARTH_MOON], SCALE_FACTOR_EARTH_MOON, true);
		ShowEarthViewport();
	}
}

void AnimateMars()
{
	static float exp = 0.0f;
	static GLfloat fTranslateZ = 0.0f;

	GLfloat x = gpEllipsePathMars->EllipseData.fSemiMajorAxis * (GLfloat)cos(PickingData.gfAngleTranslate);
	GLfloat z = -gpEllipsePathMars->EllipseData.fSemiMinorAxis * (GLfloat)sin(PickingData.gfAngleTranslate);

	POINT_TRACE p1 = { x, 0.0f, z };
	POINT_TRACE p2 = { ANIMATION_CAMERA_X, ANIMATION_CAMERA_Y, ANIMATION_CAMERA_Z };

	POINT_TRACE p = GetPointOnLine(p1, p2, z + fTranslateZ);

	if (false == gbIsPickTransitionDone)
	{
		if (p.z < ANIMATION_CAMERA_Z)
		{
			fTranslateZ = fTranslateZ + exp;
			exp = exp + TRANSLATION_EXP_INCREATE;
		}
		else
		{
			fTranslateZ = 0.0f;
			exp = 0.0f;
			gbIsPickTransitionDone = true;
		}

		glUseProgram(BasicPlanetShader.ShaderObject.uiShaderProgramObject);

		vmath::mat4 ModelMatrix = vmath::mat4::identity();
		vmath::mat4 ViewMatrix = vmath::mat4::identity();
		vmath::mat4 TranslationMatrix = vmath::mat4::identity();
		vmath::mat4 RotationMatrix = vmath::mat4::identity();
		vmath::mat4 ScaleMatrix = vmath::mat4::identity();

		TranslationMatrix = vmath::translate(p.x, p.y, p.z);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		RotationMatrix = vmath::rotate(-90.0f, 1.0f, 0.0f, 0.0f); // after this y becomes -z and z becomes y
		RotationMatrix = RotationMatrix * vmath::rotate(gfAngle, 0.0f, 0.0f, 1.0f);
		ModelMatrix = ModelMatrix * RotationMatrix;

		ScaleMatrix = vmath::scale(SCALE_FACTOR_MARS);
		ModelMatrix = ModelMatrix * ScaleMatrix;

		glUniformMatrix4fv(BasicPlanetShader.uiModelMatrixUniform, 1, GL_FALSE, ModelMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiViewMatrixUniform, 1, GL_FALSE, gmat4ViewMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, guiTexturePlanets[PLANET_AND_MOONS::MARS]);
		glUniform1i(BasicPlanetShader.uiTextureSamplerUniform, 0);

		glUniform3fv(BasicPlanetShader.uiLaUniform, 1, gfAmbientLight);
		glUniform3fv(BasicPlanetShader.uiLdUniform, 1, gfDiffuseLight);
		glUniform3fv(BasicPlanetShader.uiLsUniform, 1, gfSpecularLight);
		glUniform4fv(BasicPlanetShader.uiLightPositionUniform, 1, gfLightPosition);

		glUniform3fv(BasicPlanetShader.uiKaUniform, 1, gfAmbientMaterial);
		glUniform3fv(BasicPlanetShader.uiKdUniform, 1, gfDiffuseMaterial);
		glUniform3fv(BasicPlanetShader.uiKsUniform, 1, gfSpecularMaterial);
		glUniform1f(BasicPlanetShader.uiMaterialShininessUniform, gfMaterialShininess);

		DrawPlanet(&Planet);

		//moon
		ModelMatrix = vmath::mat4::identity();
		TranslationMatrix = vmath::mat4::identity();
		RotationMatrix = vmath::mat4::identity();
		ScaleMatrix = vmath::mat4::identity();

		gfMoonTranslationX = (gfSunRadius * SCALE_FACTOR_MARS + gfMoonRadius + 0.5f) * (GLfloat)cos(gfMoonAngleTranslate);
		gfMoonTranslationZ = (gfSunRadius * SCALE_FACTOR_MARS + gfMoonRadius + 0.5f) * (GLfloat)sin(gfMoonAngleTranslate);

		TranslationMatrix = vmath::mat4::identity();
		TranslationMatrix = vmath::translate(p.x, p.y, p.z);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		TranslationMatrix = vmath::mat4::identity();
		TranslationMatrix = vmath::translate(gfMoonTranslationX, 0.0f, -gfMoonTranslationZ);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		RotationMatrix = vmath::rotate(-90.0f, 1.0f, 0.0f, 0.0f);
		RotationMatrix = RotationMatrix * vmath::rotate(gfMoonAngle, 0.0f, 0.0f, 1.0f);
		ModelMatrix = ModelMatrix * RotationMatrix;

		ScaleMatrix = vmath::scale(SCALE_FACTOR_MARS_PHOBOS);
		ModelMatrix = ModelMatrix * ScaleMatrix;

		glUniformMatrix4fv(BasicPlanetShader.uiModelMatrixUniform, 1, GL_FALSE, ModelMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiViewMatrixUniform, 1, GL_FALSE, gmat4ViewMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, guiTexturePlanets[PLANET_AND_MOONS::EARTH_MOON]);
		glUniform1i(BasicPlanetShader.uiTextureSamplerUniform, 0);

		DrawPlanet(&SphereMoon);

		glUseProgram(0);
	}
	else
	{
		DrawNoiseMarble(SCALE_FACTOR_EARTH, SCALE_FACTOR_MARS_PHOBOS, true, vmath::vec3(0.0f, 0.0f, 0.0f), vmath::vec3(0.0f, 0.0f, 0.0f));
		//RenderPlanetOntoABorderedViewport(guiTexturePlanets[PLANET_AND_MOONS::MARS], SCALE_FACTOR_MARS, guiTexturePlanets[PLANET_AND_MOONS::EARTH_MOON], SCALE_FACTOR_MARS_PHOBOS, true);
		ShowMarsViewport();
	}
}

void AnimateJupiter()
{
	static float exp = 0.0f;
	static GLfloat fTranslateZ = 0.0f;

	GLfloat x = gpEllipsePathJupiter->EllipseData.fSemiMajorAxis * (GLfloat)cos(PickingData.gfAngleTranslate);
	GLfloat z = -gpEllipsePathJupiter->EllipseData.fSemiMinorAxis * (GLfloat)sin(PickingData.gfAngleTranslate);

	POINT_TRACE p1 = { x, 0.0f, z };
	POINT_TRACE p2 = { ANIMATION_CAMERA_X, ANIMATION_CAMERA_Y, ANIMATION_CAMERA_Z };

	POINT_TRACE p = GetPointOnLine(p1, p2, z + fTranslateZ);

	if (false == gbIsPickTransitionDone)
	{
		if (p.z < ANIMATION_CAMERA_Z)
		{
			fTranslateZ = fTranslateZ + exp;
			exp = exp + TRANSLATION_EXP_INCREATE;
		}
		else
		{
			fTranslateZ = 0.0f;
			exp = 0.0f;
			gbIsPickTransitionDone = true;
		}

		glUseProgram(BasicPlanetShader.ShaderObject.uiShaderProgramObject);

		vmath::mat4 ModelMatrix = vmath::mat4::identity();
		vmath::mat4 ViewMatrix = vmath::mat4::identity();
		vmath::mat4 TranslationMatrix = vmath::mat4::identity();
		vmath::mat4 RotationMatrix = vmath::mat4::identity();
		vmath::mat4 ScaleMatrix = vmath::mat4::identity();

		TranslationMatrix = vmath::translate(p.x, p.y, p.z);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		RotationMatrix = vmath::rotate(-90.0f, 1.0f, 0.0f, 0.0f); // after this y becomes -z and z becomes y
		RotationMatrix = RotationMatrix * vmath::rotate(gfAngle, 0.0f, 0.0f, 1.0f);
		ModelMatrix = ModelMatrix * RotationMatrix;

		ScaleMatrix = vmath::scale(SCALE_FACTOR_JUPITER);
		ModelMatrix = ModelMatrix * ScaleMatrix;

		glUniformMatrix4fv(BasicPlanetShader.uiModelMatrixUniform, 1, GL_FALSE, ModelMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiViewMatrixUniform, 1, GL_FALSE, gmat4ViewMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, guiTexturePlanets[PLANET_AND_MOONS::JUPITER]);
		glUniform1i(BasicPlanetShader.uiTextureSamplerUniform, 0);

		glUniform3fv(BasicPlanetShader.uiLaUniform, 1, gfAmbientLight);
		glUniform3fv(BasicPlanetShader.uiLdUniform, 1, gfDiffuseLight);
		glUniform3fv(BasicPlanetShader.uiLsUniform, 1, gfSpecularLight);
		glUniform4fv(BasicPlanetShader.uiLightPositionUniform, 1, gfLightPosition);

		glUniform3fv(BasicPlanetShader.uiKaUniform, 1, gfAmbientMaterial);
		glUniform3fv(BasicPlanetShader.uiKdUniform, 1, gfDiffuseMaterial);
		glUniform3fv(BasicPlanetShader.uiKsUniform, 1, gfSpecularMaterial);
		glUniform1f(BasicPlanetShader.uiMaterialShininessUniform, gfMaterialShininess);
		
		DrawPlanet(&Planet);

		//moon
		ModelMatrix = vmath::mat4::identity();
		TranslationMatrix = vmath::mat4::identity();
		RotationMatrix = vmath::mat4::identity();
		ScaleMatrix = vmath::mat4::identity();

		gfMoonTranslationX = (gfSunRadius * SCALE_FACTOR_JUPITER + gfMoonRadius + 0.5f) * (GLfloat)cos(gfMoonAngleTranslate);
		gfMoonTranslationZ = (gfSunRadius * SCALE_FACTOR_JUPITER + gfMoonRadius + 0.5f) * (GLfloat)sin(gfMoonAngleTranslate);

		TranslationMatrix = vmath::mat4::identity();
		TranslationMatrix = vmath::translate(p.x, p.y, p.z);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		TranslationMatrix = vmath::mat4::identity();
		TranslationMatrix = vmath::translate(gfMoonTranslationX, 0.0f, -gfMoonTranslationZ);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		RotationMatrix = vmath::rotate(-90.0f, 1.0f, 0.0f, 0.0f);
		RotationMatrix = RotationMatrix * vmath::rotate(gfMoonAngle, 0.0f, 0.0f, 1.0f);
		ModelMatrix = ModelMatrix * RotationMatrix;

		ScaleMatrix = vmath::scale(SCALE_FACTOR_JUPITER_EUROPA);
		ModelMatrix = ModelMatrix * ScaleMatrix;

		glUniformMatrix4fv(BasicPlanetShader.uiModelMatrixUniform, 1, GL_FALSE, ModelMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiViewMatrixUniform, 1, GL_FALSE, gmat4ViewMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, guiTexturePlanets[PLANET_AND_MOONS::EARTH_MOON]);
		glUniform1i(BasicPlanetShader.uiTextureSamplerUniform, 0);

		DrawPlanet(&SphereMoon);

		glUseProgram(0);
	}
	else
	{
		DrawNoiseMarble(1.0f, SCALE_FACTOR_JUPITER_EUROPA, true, vmath::vec3(0.0f, 0.0f, 0.0f), vmath::vec3(0.0f, 0.0f, 0.0f));
		//RenderPlanetOntoABorderedViewport(guiTexturePlanets[PLANET_AND_MOONS::JUPITER], 1.0f, guiTexturePlanets[PLANET_AND_MOONS::EARTH_MOON], SCALE_FACTOR_JUPITER_EUROPA, true);
		ShowJupiterViewport();
	}
}

void AnimateSaturn()
{
	static float exp = 0.0f;
	static GLfloat fTranslateZ = 0.0f;

	GLfloat x = gpEllipsePathSaturn->EllipseData.fSemiMajorAxis * (GLfloat)cos(PickingData.gfAngleTranslate);
	GLfloat z = -gpEllipsePathSaturn->EllipseData.fSemiMinorAxis * (GLfloat)sin(PickingData.gfAngleTranslate);

	POINT_TRACE p1 = { x, 0.0f, z };
	POINT_TRACE p2 = { ANIMATION_CAMERA_X, ANIMATION_CAMERA_Y, ANIMATION_CAMERA_Z };

	POINT_TRACE p = GetPointOnLine(p1, p2, z + fTranslateZ);

	if (false == gbIsPickTransitionDone)
	{
		if (p.z < ANIMATION_CAMERA_Z)
		{
			fTranslateZ = fTranslateZ + exp;
			exp = exp + TRANSLATION_EXP_INCREATE;
		}
		else
		{
			fTranslateZ = 0.0f;
			exp = 0.0f;
			gbIsPickTransitionDone = true;
		}

		glUseProgram(BasicPlanetShader.ShaderObject.uiShaderProgramObject);

		vmath::mat4 ModelMatrix = vmath::mat4::identity();
		vmath::mat4 ViewMatrix = vmath::mat4::identity();
		vmath::mat4 TranslationMatrix = vmath::mat4::identity();
		vmath::mat4 RotationMatrix = vmath::mat4::identity();
		vmath::mat4 ScaleMatrix = vmath::mat4::identity();

		TranslationMatrix = vmath::translate(p.x, p.y, p.z);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		RotationMatrix = vmath::rotate(-90.0f, 1.0f, 0.0f, 0.0f); // after this y becomes -z and z becomes y
		RotationMatrix = RotationMatrix * vmath::rotate(gfAngle, 0.0f, 0.0f, 1.0f);
		ModelMatrix = ModelMatrix * RotationMatrix;

		ScaleMatrix = vmath::scale(SCALE_FACTOR_SATURN);
		ModelMatrix = ModelMatrix * ScaleMatrix;

		glUniformMatrix4fv(BasicPlanetShader.uiModelMatrixUniform, 1, GL_FALSE, ModelMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiViewMatrixUniform, 1, GL_FALSE, gmat4ViewMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, guiTexturePlanets[PLANET_AND_MOONS::SATURN]);
		glUniform1i(BasicPlanetShader.uiTextureSamplerUniform, 0);

		glUniform3fv(BasicPlanetShader.uiLaUniform, 1, gfAmbientLight);
		glUniform3fv(BasicPlanetShader.uiLdUniform, 1, gfDiffuseLight);
		glUniform3fv(BasicPlanetShader.uiLsUniform, 1, gfSpecularLight);
		glUniform4fv(BasicPlanetShader.uiLightPositionUniform, 1, gfLightPosition);

		glUniform3fv(BasicPlanetShader.uiKaUniform, 1, gfAmbientMaterial);
		glUniform3fv(BasicPlanetShader.uiKdUniform, 1, gfDiffuseMaterial);
		glUniform3fv(BasicPlanetShader.uiKsUniform, 1, gfSpecularMaterial);
		glUniform1f(BasicPlanetShader.uiMaterialShininessUniform, gfMaterialShininess);

		DrawPlanet(&Planet);

		ModelMatrix = vmath::mat4::identity();
		TranslationMatrix = vmath::translate(p.x, p.y, p.z);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		ScaleMatrix = vmath::scale(SCALE_FACTOR_SATURN);
		ModelMatrix = ModelMatrix * ScaleMatrix;

		RotationMatrix = vmath::mat4::identity();
		RotationMatrix = vmath::rotate(-90.0f, 1.0f, 0.0f, 0.0f); // after this y becomes -z and z becomes y
		RotationMatrix = vmath::rotate(-80.0f, vmath::vec3(1.0f, 0.0f, 0.0f));
		ModelMatrix = ModelMatrix * RotationMatrix;

		glUniformMatrix4fv(BasicPlanetShader.uiModelMatrixUniform, 1, GL_FALSE, ModelMatrix);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, guiTexturePlanets[PLANET_AND_MOONS::SATURN_RING_Tex]);
		glUniform1i(BasicPlanetShader.uiTextureSamplerUniform, 0);

		DrawSaturnRing(&SaturnRing);

		//moon
		ModelMatrix = vmath::mat4::identity();
		TranslationMatrix = vmath::mat4::identity();
		RotationMatrix = vmath::mat4::identity();
		ScaleMatrix = vmath::mat4::identity();

		gfMoonTranslationX = (gfSunRadius * SCALE_FACTOR_SATURN + gfMoonRadius + 0.5f) * (GLfloat)cos(gfMoonAngleTranslate);
		gfMoonTranslationZ = (gfSunRadius * SCALE_FACTOR_SATURN + gfMoonRadius + 0.5f) * (GLfloat)sin(gfMoonAngleTranslate);

		TranslationMatrix = vmath::mat4::identity();
		TranslationMatrix = vmath::translate(p.x, p.y, p.z);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		TranslationMatrix = vmath::mat4::identity();
		TranslationMatrix = vmath::translate(gfMoonTranslationX, 2.0f, -gfMoonTranslationZ);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		RotationMatrix = vmath::rotate(-90.0f, 1.0f, 0.0f, 0.0f);
		RotationMatrix = RotationMatrix * vmath::rotate(gfMoonAngle, 0.0f, 0.0f, 1.0f);
		ModelMatrix = ModelMatrix * RotationMatrix;

		ScaleMatrix = vmath::scale(SCALE_FACTOR_SATURN_TITAN);
		ModelMatrix = ModelMatrix * ScaleMatrix;

		glUniformMatrix4fv(BasicPlanetShader.uiModelMatrixUniform, 1, GL_FALSE, ModelMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiViewMatrixUniform, 1, GL_FALSE, gmat4ViewMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, guiTexturePlanets[PLANET_AND_MOONS::EARTH_MOON]);
		glUniform1i(BasicPlanetShader.uiTextureSamplerUniform, 0);

		DrawPlanet(&SphereMoon);

		glUseProgram(0);
	}
	else
	{
		DrawNoiseMarble(1.0f, SCALE_FACTOR_SATURN_TITAN, true, vmath::vec3(0.0f, 0.0f, 0.0f), vmath::vec3(0.0f, 0.0f, 0.0f));
		//RenderPlanetOntoABorderedViewport(guiTexturePlanets[PLANET_AND_MOONS::SATURN], 1.0f, guiTexturePlanets[PLANET_AND_MOONS::EARTH_MOON], SCALE_FACTOR_SATURN_TITAN, true);
		ShowSaturnViewport();
	}
}

void AnimateUranus()
{
	static float exp = 0.0f;
	static GLfloat fTranslateZ = 0.0f;

	GLfloat x = gpEllipsePathUranus->EllipseData.fSemiMajorAxis * (GLfloat)cos(PickingData.gfAngleTranslate);
	GLfloat z = -gpEllipsePathUranus->EllipseData.fSemiMinorAxis * (GLfloat)sin(PickingData.gfAngleTranslate);

	POINT_TRACE p1 = { x, 0.0f, z };
	POINT_TRACE p2 = { ANIMATION_CAMERA_X, ANIMATION_CAMERA_Y, ANIMATION_CAMERA_Z };

	POINT_TRACE p = GetPointOnLine(p1, p2, z + fTranslateZ);

	if (false == gbIsPickTransitionDone)
	{
		if (p.z < ANIMATION_CAMERA_Z)
		{
			fTranslateZ = fTranslateZ + exp;
			exp = exp + TRANSLATION_EXP_INCREATE;
		}
		else
		{
			fTranslateZ = 0.0f;
			exp = 0.0f;
			gbIsPickTransitionDone = true;
		}

		glUseProgram(BasicPlanetShader.ShaderObject.uiShaderProgramObject);

		vmath::mat4 ModelMatrix = vmath::mat4::identity();
		vmath::mat4 ViewMatrix = vmath::mat4::identity();
		vmath::mat4 TranslationMatrix = vmath::mat4::identity();
		vmath::mat4 RotationMatrix = vmath::mat4::identity();
		vmath::mat4 ScaleMatrix = vmath::mat4::identity();

		TranslationMatrix = vmath::translate(p.x, p.y, p.z);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		RotationMatrix = vmath::rotate(-90.0f, 1.0f, 0.0f, 0.0f); // after this y becomes -z and z becomes y
		RotationMatrix = RotationMatrix * vmath::rotate(gfAngle, 0.0f, 0.0f, 1.0f);
		ModelMatrix = ModelMatrix * RotationMatrix;

		ScaleMatrix = vmath::scale(SCALE_FACTOR_URANUS);
		ModelMatrix = ModelMatrix * ScaleMatrix;

		glUniformMatrix4fv(BasicPlanetShader.uiModelMatrixUniform, 1, GL_FALSE, ModelMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiViewMatrixUniform, 1, GL_FALSE, gmat4ViewMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, guiTexturePlanets[PLANET_AND_MOONS::URANUS]);
		glUniform1i(BasicPlanetShader.uiTextureSamplerUniform, 0);

		glUniform3fv(BasicPlanetShader.uiLaUniform, 1, gfAmbientLight);
		glUniform3fv(BasicPlanetShader.uiLdUniform, 1, gfDiffuseLight);
		glUniform3fv(BasicPlanetShader.uiLsUniform, 1, gfSpecularLight);
		glUniform4fv(BasicPlanetShader.uiLightPositionUniform, 1, gfLightPosition);

		glUniform3fv(BasicPlanetShader.uiKaUniform, 1, gfAmbientMaterial);
		glUniform3fv(BasicPlanetShader.uiKdUniform, 1, gfDiffuseMaterial);
		glUniform3fv(BasicPlanetShader.uiKsUniform, 1, gfSpecularMaterial);
		glUniform1f(BasicPlanetShader.uiMaterialShininessUniform, gfMaterialShininess);

		DrawPlanet(&Planet);

		//moon
		ModelMatrix = vmath::mat4::identity();
		TranslationMatrix = vmath::mat4::identity();
		RotationMatrix = vmath::mat4::identity();
		ScaleMatrix = vmath::mat4::identity();

		gfMoonTranslationX = (gfSunRadius * SCALE_FACTOR_URANUS + gfMoonRadius + 0.5f) * (GLfloat)cos(gfMoonAngleTranslate);
		gfMoonTranslationZ = (gfSunRadius * SCALE_FACTOR_URANUS + gfMoonRadius + 0.5f) * (GLfloat)sin(gfMoonAngleTranslate);

		TranslationMatrix = vmath::mat4::identity();
		TranslationMatrix = vmath::translate(p.x, p.y, p.z);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		TranslationMatrix = vmath::mat4::identity();
		TranslationMatrix = vmath::translate(gfMoonTranslationX, 0.0f, -gfMoonTranslationZ);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		RotationMatrix = vmath::rotate(-90.0f, 1.0f, 0.0f, 0.0f);
		RotationMatrix = RotationMatrix * vmath::rotate(gfMoonAngle, 0.0f, 0.0f, 1.0f);
		ModelMatrix = ModelMatrix * RotationMatrix;

		ScaleMatrix = vmath::scale(SCALE_FACTOR_URANUS_AERIAL);
		ModelMatrix = ModelMatrix * ScaleMatrix;

		glUniformMatrix4fv(BasicPlanetShader.uiModelMatrixUniform, 1, GL_FALSE, ModelMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiViewMatrixUniform, 1, GL_FALSE, gmat4ViewMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, guiTexturePlanets[PLANET_AND_MOONS::EARTH_MOON]);
		glUniform1i(BasicPlanetShader.uiTextureSamplerUniform, 0);

		DrawPlanet(&SphereMoon);

		glUseProgram(0);
	}
	else
	{
		DrawNoiseMarble(1.0f, SCALE_FACTOR_URANUS_AERIAL, true, vmath::vec3(0.0f, 0.0f, 0.0f), vmath::vec3(0.0f, 0.0f, 0.0f));
		//RenderPlanetOntoABorderedViewport(guiTexturePlanets[PLANET_AND_MOONS::URANUS], 1.0f, guiTexturePlanets[PLANET_AND_MOONS::EARTH_MOON], SCALE_FACTOR_URANUS_AERIAL, true);
		ShowUranusViewport();
	}
}

void AnimateNeptune()
{
	static float exp = 0.0f;
	static GLfloat fTranslateZ = 0.0f;

	GLfloat x = gpEllipsePathNeptune->EllipseData.fSemiMajorAxis * (GLfloat)cos(PickingData.gfAngleTranslate);
	GLfloat z = -gpEllipsePathNeptune->EllipseData.fSemiMinorAxis * (GLfloat)sin(PickingData.gfAngleTranslate);

	POINT_TRACE p1 = { x, 0.0f, z };
	POINT_TRACE p2 = { ANIMATION_CAMERA_X, ANIMATION_CAMERA_Y, ANIMATION_CAMERA_Z };

	POINT_TRACE p = GetPointOnLine(p1, p2, z + fTranslateZ);

	if (false == gbIsPickTransitionDone)
	{
		if (p.z < ANIMATION_CAMERA_Z)
		{
			fTranslateZ = fTranslateZ + exp;
			exp = exp + TRANSLATION_EXP_INCREATE;
		}
		else
		{
			fTranslateZ = 0.0f;
			exp = 0.0f;
			gbIsPickTransitionDone = true;
		}

		glUseProgram(BasicPlanetShader.ShaderObject.uiShaderProgramObject);

		vmath::mat4 ModelMatrix = vmath::mat4::identity();
		vmath::mat4 ViewMatrix = vmath::mat4::identity();
		vmath::mat4 TranslationMatrix = vmath::mat4::identity();
		vmath::mat4 RotationMatrix = vmath::mat4::identity();
		vmath::mat4 ScaleMatrix = vmath::mat4::identity();

		TranslationMatrix = vmath::translate(p.x, p.y, p.z);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		RotationMatrix = vmath::rotate(-90.0f, 1.0f, 0.0f, 0.0f); // after this y becomes -z and z becomes y
		RotationMatrix = RotationMatrix * vmath::rotate(gfAngle, 0.0f, 0.0f, 1.0f);
		ModelMatrix = ModelMatrix * RotationMatrix;

		ScaleMatrix = vmath::scale(SCALE_FACTOR_NEPTUNE);
		ModelMatrix = ModelMatrix * ScaleMatrix;

		glUniformMatrix4fv(BasicPlanetShader.uiModelMatrixUniform, 1, GL_FALSE, ModelMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiViewMatrixUniform, 1, GL_FALSE, gmat4ViewMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, guiTexturePlanets[PLANET_AND_MOONS::NEPTUNE]);
		glUniform1i(BasicPlanetShader.uiTextureSamplerUniform, 0);

		glUniform3fv(BasicPlanetShader.uiLaUniform, 1, gfAmbientLight);
		glUniform3fv(BasicPlanetShader.uiLdUniform, 1, gfDiffuseLight);
		glUniform3fv(BasicPlanetShader.uiLsUniform, 1, gfSpecularLight);
		glUniform4fv(BasicPlanetShader.uiLightPositionUniform, 1, gfLightPosition);

		glUniform3fv(BasicPlanetShader.uiKaUniform, 1, gfAmbientMaterial);
		glUniform3fv(BasicPlanetShader.uiKdUniform, 1, gfDiffuseMaterial);
		glUniform3fv(BasicPlanetShader.uiKsUniform, 1, gfSpecularMaterial);
		glUniform1f(BasicPlanetShader.uiMaterialShininessUniform, gfMaterialShininess);

		DrawPlanet(&Planet);

		//moon
		ModelMatrix = vmath::mat4::identity();
		TranslationMatrix = vmath::mat4::identity();
		RotationMatrix = vmath::mat4::identity();
		ScaleMatrix = vmath::mat4::identity();

		gfMoonTranslationX = (gfSunRadius * SCALE_FACTOR_NEPTUNE + gfMoonRadius + 0.5f) * (GLfloat)cos(gfMoonAngleTranslate);
		gfMoonTranslationZ = (gfSunRadius * SCALE_FACTOR_NEPTUNE + gfMoonRadius + 0.5f) * (GLfloat)sin(gfMoonAngleTranslate);

		TranslationMatrix = vmath::mat4::identity();
		TranslationMatrix = vmath::translate(p.x, p.y, p.z);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		TranslationMatrix = vmath::mat4::identity();
		TranslationMatrix = vmath::translate(gfMoonTranslationX, 0.0f, -gfMoonTranslationZ);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		RotationMatrix = vmath::rotate(-90.0f, 1.0f, 0.0f, 0.0f);
		RotationMatrix = RotationMatrix * vmath::rotate(gfMoonAngle, 0.0f, 0.0f, 1.0f);
		ModelMatrix = ModelMatrix * RotationMatrix;

		ScaleMatrix = vmath::scale(SCALE_FACTOR_NEPTUNE_TRITON);
		ModelMatrix = ModelMatrix * ScaleMatrix;

		glUniformMatrix4fv(BasicPlanetShader.uiModelMatrixUniform, 1, GL_FALSE, ModelMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiViewMatrixUniform, 1, GL_FALSE, gmat4ViewMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, guiTexturePlanets[PLANET_AND_MOONS::EARTH_MOON]);
		glUniform1i(BasicPlanetShader.uiTextureSamplerUniform, 0);

		DrawPlanet(&SphereMoon);

		glUseProgram(0);
	}
	else
	{
		DrawNoiseMarble(1.0f, SCALE_FACTOR_NEPTUNE_TRITON, true, vmath::vec3(0.0f, 0.0f, 0.0f), vmath::vec3(0.0f, 0.0f, 0.0f));
		//RenderPlanetOntoABorderedViewport(guiTexturePlanets[PLANET_AND_MOONS::NEPTUNE], 1.0f, guiTexturePlanets[PLANET_AND_MOONS::EARTH_MOON], SCALE_FACTOR_NEPTUNE_TRITON, true);
		ShowNeptuneViewport();
	}
}

void AnimatePluto()
{
	static float exp = 0.0f;
	static GLfloat fTranslateZ = 0.0f;

	GLfloat x = gpEllipsePathPluto->EllipseData.fSemiMajorAxis * (GLfloat)cos(PickingData.gfAngleTranslate);
	GLfloat z = -gpEllipsePathPluto->EllipseData.fSemiMinorAxis * (GLfloat)sin(PickingData.gfAngleTranslate);

	POINT_TRACE p1 = { x, 0.0f, z };
	POINT_TRACE p2 = { ANIMATION_CAMERA_X, ANIMATION_CAMERA_Y, ANIMATION_CAMERA_Z };

	POINT_TRACE p = GetPointOnLine(p1, p2, z + fTranslateZ);

	if (false == gbIsPickTransitionDone)
	{
		if (p.z < ANIMATION_CAMERA_Z)
		{
			fTranslateZ = fTranslateZ + exp;
			exp = exp + TRANSLATION_EXP_INCREATE;
		}
		else
		{
			fTranslateZ = 0.0f;
			exp = 0.0f;
			gbIsPickTransitionDone = true;
		}

		glUseProgram(BasicPlanetShader.ShaderObject.uiShaderProgramObject);

		vmath::mat4 ModelMatrix = vmath::mat4::identity();
		vmath::mat4 ViewMatrix = vmath::mat4::identity();
		vmath::mat4 TranslationMatrix = vmath::mat4::identity();
		vmath::mat4 RotationMatrix = vmath::mat4::identity();
		vmath::mat4 ScaleMatrix = vmath::mat4::identity();

		TranslationMatrix = vmath::translate(p.x, p.y, p.z);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		RotationMatrix = vmath::rotate(-90.0f, 1.0f, 0.0f, 0.0f); // after this y becomes -z and z becomes y
		RotationMatrix = RotationMatrix * vmath::rotate(gfAngle, 0.0f, 0.0f, 1.0f);
		ModelMatrix = ModelMatrix * RotationMatrix;

		ScaleMatrix = vmath::scale(SCALE_FACTOR_PLUTO);
		ModelMatrix = ModelMatrix * ScaleMatrix;

		glUniformMatrix4fv(BasicPlanetShader.uiModelMatrixUniform, 1, GL_FALSE, ModelMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiViewMatrixUniform, 1, GL_FALSE, gmat4ViewMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, guiTexturePlanets[PLANET_AND_MOONS::PLUTO]);
		glUniform1i(BasicPlanetShader.uiTextureSamplerUniform, 0);

		glUniform3fv(BasicPlanetShader.uiLaUniform, 1, gfAmbientLight);
		glUniform3fv(BasicPlanetShader.uiLdUniform, 1, gfDiffuseLight);
		glUniform3fv(BasicPlanetShader.uiLsUniform, 1, gfSpecularLight);
		glUniform4fv(BasicPlanetShader.uiLightPositionUniform, 1, gfLightPosition);

		glUniform3fv(BasicPlanetShader.uiKaUniform, 1, gfAmbientMaterial);
		glUniform3fv(BasicPlanetShader.uiKdUniform, 1, gfDiffuseMaterial);
		glUniform3fv(BasicPlanetShader.uiKsUniform, 1, gfSpecularMaterial);
		glUniform1f(BasicPlanetShader.uiMaterialShininessUniform, gfMaterialShininess);

		DrawPlanet(&Planet);

		glUseProgram(0);
	}
	else
	{
		RenderPlanetOntoABorderedViewport(guiTexturePlanets[PLANET_AND_MOONS::PLUTO], 1.0f, 0, 0.0f, false);
		ShowPlutoViewport();
	}
}

void FillPickingData(PLANET_AND_MOONS pick)
{
	switch (pick)
	{
	case PLANET_AND_MOONS::MERCURY:
		PickingData.gfAngleTranslate = gfMercuryAngleTranslate;
		break;
	case PLANET_AND_MOONS::VENUS:
		PickingData.gfAngleTranslate = gfVenusAngleTranslate;
		break;
	case PLANET_AND_MOONS::EARTH:
		PickingData.gfAngleTranslate = gfEarthAngleTranslate;
		break;
	case PLANET_AND_MOONS::MARS:
		PickingData.gfAngleTranslate = gfMarsAngleTranslate;
		break;
	case PLANET_AND_MOONS::JUPITER:
		PickingData.gfAngleTranslate = gfJupiterAngleTranslate;
		break;
	case PLANET_AND_MOONS::SATURN:
		PickingData.gfAngleTranslate = gfSaturnAngleTranslate;
		break;
	case PLANET_AND_MOONS::URANUS:
		PickingData.gfAngleTranslate = gfUranusAngleTranslate;
		break;
	case PLANET_AND_MOONS::NEPTUNE:
		PickingData.gfAngleTranslate = gfNeptuneAngleTranslate;
		break;
	case PLANET_AND_MOONS::PLUTO:
		PickingData.gfAngleTranslate = gfPlutoAngleTranslate;
		break;
	default:
		PickingData.gfAngleTranslate = 0.0f;
		break;
	}
}

void ClearPickingData()
{
	PickingData.gfAngleTranslate = 0.0f;
}

POINT_TRACE GetPointOnLine(POINT_TRACE p1, POINT_TRACE p2, float z)
{
	POINT_TRACE p = { 0 };

	p.x = (((p2.x - p1.x) / (p2.z - p1.z)) * (z - p1.z)) + p1.x;
	p.y = (((p2.y - p1.y) / (p2.z - p1.z)) * (z - p1.z)) + p1.y;
	p.z = z;

	return p;
}

void ShowSunViewport()
{
	DrawBorderedViewport(gWindowWidth - 450, 0, 450, 250);
	
	vmath::vec4 color(0.0f, 1.0f, 0.0f, 1.0f);
	RenderFont(&FontShader, FontMapArial, "SUN", color, -30, 45, 0.4f);
	RenderFont(&FontShader, FontMapArial, "It Is The Oldeset And Biggest Star Of Our Solar System.", color, -165, 10, 0.2f);
	RenderFont(&FontShader, FontMapArial, "Its Diameter Is About 1.39 Million Kilometres.", color, -165, -10, 0.2f);
	RenderFont(&FontShader, FontMapArial, "It Consists of Hydrogen, Helium, Oxygen, Carbon, Neon", color, -165, -30, 0.2f);
	RenderFont(&FontShader, FontMapArial, "   and Iron.", color, -165, -50, 0.2f);
	RenderFont(&FontShader, FontMapArial, "NO Life Is Possible Without Sun.", color, -165, -70, 0.2f);

	glViewport(0, 0, gWindowWidth, gWindowHeight);
}

void ShowMercuryViewport()
{
	DrawBorderedViewport(gWindowWidth - 450, 0, 450, 250);

	vmath::vec4 color(0.0f, 1.0f, 0.0f, 1.0f);
	RenderFont(&FontShader, FontMapArial, "MERCURY", color, -40, 50, 0.4f);
	RenderFont(&FontShader, FontMapArial, "It Is The First And Smallest Planet In Solar System.", color, -165, 10, 0.2f);
	RenderFont(&FontShader, FontMapArial, "It Takes 87.97 Days To Orbit Around The Sun.", color, -165, -10, 0.2f);
	RenderFont(&FontShader, FontMapArial, "Mercury Has Almost No Atmosphere and Hence Lesser Hot", color, -165, -30, 0.2f);
	RenderFont(&FontShader, FontMapArial, "   than Venus, The Second Planet.", color, -165, -50, 0.2f);
	RenderFont(&FontShader, FontMapArial, "Two Spacecraft Have Visisted, Mariner 10 And MESSENGER.", color, -165, -70, 0.2f);

	glViewport(0, 0, gWindowWidth, gWindowHeight);
}

void ShowVenusViewport()
{
	DrawBorderedViewport(gWindowWidth - 450, 0, 450, 250);

	vmath::vec4 color(0.0f, 1.0f, 0.0f, 1.0f);
	RenderFont(&FontShader, FontMapArial, "VENUS", color, -48, 50, 0.4f);
	RenderFont(&FontShader, FontMapArial, "It Is The Brightest Natural Object In Earth's Night Sky", color, -165, 10, 0.2f);
	RenderFont(&FontShader, FontMapArial, "   After Its Moon.", color, -165, -10, 0.2f);
	RenderFont(&FontShader, FontMapArial, "Its Atmosphere Consists Of More Than 96% Of CO2", color, -165, -30, 0.2f);
	RenderFont(&FontShader, FontMapArial, "Venus Has The Hottest Surface Of Any Planet.", color, -165, -50, 0.2f);
	RenderFont(&FontShader, FontMapArial, "It Is Sometimes Called Earth's 'Sister Planet'.", color, -165, -70, 0.2f);

	glViewport(0, 0, gWindowWidth, gWindowHeight);
}

void ShowEarthViewport()
{
	DrawBorderedViewport(gWindowWidth - 450, 0, 450, 250);

	vmath::vec4 color(0.0f, 1.0f, 0.0f, 1.0f);
	RenderFont(&FontShader, FontMapArial, "EARTH", color, -38, 50, 0.4f);
	RenderFont(&FontShader, FontMapArial, "Mother Earth...", color, -165, 10, 0.2f);
	RenderFont(&FontShader, FontMapArial, "It Is The Only Planet With Water And Presence Of Life.", color, -165, -10, 0.2f);
	RenderFont(&FontShader, FontMapArial, "Earth's Atmosphere Consists Mostly Of Nitrogen And Oxygen.", color, -165, -30, 0.2f);
	RenderFont(&FontShader, FontMapArial, "It Takes 365 Days To Complete One Orbit Around the Sun.", color, -165, -50, 0.2f);
	RenderFont(&FontShader, FontMapArial, "Earth Is The Densest Planet In The Solar System.", color, -165, -70, 0.2f);

	glViewport(0, 0, gWindowWidth, gWindowHeight);
}

void ShowMarsViewport()
{
	DrawBorderedViewport(gWindowWidth - 450, 0, 450, 250);

	vmath::vec4 color(0.0f, 1.0f, 0.0f, 1.0f);
	RenderFont(&FontShader, FontMapArial, "MARS", color, -32, 50, 0.4f);
	RenderFont(&FontShader, FontMapArial, "It Is The 4th Planet Also Known As The Red Planet.", color, -165, 10, 0.2f);
	RenderFont(&FontShader, FontMapArial, "Mars Is A Terrestial Planet With A Thin Atmosphere.", color, -165, -10, 0.2f);
	RenderFont(&FontShader, FontMapArial, "Mars Has Two Moons, Phobos and Deimos.", color, -165, -30, 0.2f);
	RenderFont(&FontShader, FontMapArial, "It Is The Only Planet With a Possibility Of Extant Life.", color, -165, -50, 0.2f);
	RenderFont(&FontShader, FontMapArial, "Mars Takes 687 Days to Complete One Orbit.", color, -165, -70, 0.2f);

	glViewport(0, 0, gWindowWidth, gWindowHeight);
}

void ShowJupiterViewport()
{
	DrawBorderedViewport(gWindowWidth - 450, 0, 450, 250);

	vmath::vec4 color(0.0f, 1.0f, 0.0f, 1.0f);
	RenderFont(&FontShader, FontMapArial, "JUPITER", color, -45, 50, 0.4f);
	RenderFont(&FontShader, FontMapArial, "It Is The Largest Planet And Biggest Gas Giant Of.", color, -165, 10, 0.2f);
	RenderFont(&FontShader, FontMapArial, "   Our Solar System.", color, -165, -10, 0.2f);
	RenderFont(&FontShader, FontMapArial, "It Is Primarily Composed Of Hydrogen And Lacks A Well-", color, -165, -30, 0.2f);
	RenderFont(&FontShader, FontMapArial, "  Define Solid Surface.", color, -165, -50, 0.2f);
	RenderFont(&FontShader, FontMapArial, "There Are 79 Known Moon, Europa Is One Of Them.", color, -165, -70, 0.2f);

	glViewport(0, 0, gWindowWidth, gWindowHeight);
}

void ShowSaturnViewport()
{
	DrawBorderedViewport(gWindowWidth - 450, 0, 450, 250);

	vmath::vec4 color(0.0f, 1.0f, 0.0f, 1.0f);
	RenderFont(&FontShader, FontMapArial, "SATURN", color, -42, 50, 0.4f);
	RenderFont(&FontShader, FontMapArial, "It Is The Second Largest Planet Of Our Solar System.", color, -165, 10, 0.2f);
	RenderFont(&FontShader, FontMapArial, "It Is One Of The Two Gas Giants Of Solar System.", color, -165, -10, 0.2f);
	RenderFont(&FontShader, FontMapArial, "It Had Total Of 82 Known Moon, Out of Which Titan", color, -165, -30, 0.2f);
	RenderFont(&FontShader, FontMapArial, "   Is The Largest Moon.", color, -165, -50, 0.2f);
	RenderFont(&FontShader, FontMapArial, "It Is The Planet With Its Unique Ring Of Moonlets.", color, -165, -70, 0.2f);

	glViewport(0, 0, gWindowWidth, gWindowHeight);
}

void ShowUranusViewport()
{
	DrawBorderedViewport(gWindowWidth - 450, 0, 450, 250);

	vmath::vec4 color(0.0f, 1.0f, 0.0f, 1.0f);
	RenderFont(&FontShader, FontMapArial, "URANUS", color, -42, 50, 0.4f);
	RenderFont(&FontShader, FontMapArial, "It Is The 3rd Largest Planet Of Our Solar System.", color, -165, 10, 0.2f);
	RenderFont(&FontShader, FontMapArial, "It Is One Of The Two Ice Giants Of Solar System.", color, -165, -10, 0.2f);
	RenderFont(&FontShader, FontMapArial, "It Takes 84 Years To Complete One Rotation Around Sun.", color, -165, -30, 0.2f);
	RenderFont(&FontShader, FontMapArial, "Uranus Has Total Of 27 Moons.", color, -165, -50, 0.2f);
	RenderFont(&FontShader, FontMapArial, "Uranus Was Discovered On 13th March 1781.", color, -165, -70, 0.2f);

	glViewport(0, 0, gWindowWidth, gWindowHeight);
}

void ShowNeptuneViewport()
{
	DrawBorderedViewport(gWindowWidth - 450, 0, 450, 250);

	vmath::vec4 color(0.0f, 1.0f, 0.0f, 1.0f);
	RenderFont(&FontShader, FontMapArial, "NEPTUNE", color, -45, 50, 0.35f);
	RenderFont(&FontShader, FontMapArial, "It Is The 4th Largest Planet Of Our Solar System.", color, -165, 10, 0.2f);
	RenderFont(&FontShader, FontMapArial, "It Is 17 Times The Mass Of The Earth.", color, -165, -10, 0.2f);
	RenderFont(&FontShader, FontMapArial, "It Takes 164.8 Years To Rotate Around The Sun", color, -165, -30, 0.2f);
	RenderFont(&FontShader, FontMapArial, "It Is The First Planet To Be Discovered By Mathematical", color, -165, -50, 0.2f);
	RenderFont(&FontShader, FontMapArial, "  Prediction Rather Than By Empirical Observation.", color, -165, -70, 0.2f);

	glViewport(0, 0, gWindowWidth, gWindowHeight);
}

void ShowPlutoViewport()
{
	DrawBorderedViewport(gWindowWidth - 450, 0, 450, 250);

	vmath::vec4 color(0.0f, 1.0f, 0.0f, 1.0f);
	RenderFont(&FontShader, FontMapArial, "PLUTO", color, -36, 50, 0.35f);
	RenderFont(&FontShader, FontMapArial, "It Is The Last Planet Of Our Solar System.", color, -165, 10, 0.2f);
	RenderFont(&FontShader, FontMapArial, "It Is Also Known As The Dwarf Planet.", color, -165, -10, 0.2f);
	RenderFont(&FontShader, FontMapArial, "Is Is Primarily Made Of Rock And Ice.", color, -165, -30, 0.2f);
	RenderFont(&FontShader, FontMapArial, "Pluto Is Yet To Complete a Full Orbit Around", color, -165, -50, 0.2f);
	RenderFont(&FontShader, FontMapArial, "   Sun Since Its Discovery.", color, -165, -70, 0.2f);

	glViewport(0, 0, gWindowWidth, gWindowHeight);
}

void DrawBorderedViewport(GLint x, GLint y, GLfloat fWidth, GLfloat fHeight)
{
	glViewport(x, y, (GLsizei)fWidth, (GLsizei)fHeight);

	GLfloat border[15] = { 0 };

	if (fWidth > fHeight)
	{
		border[0] = -1.0f * (ORTHO * ((GLfloat)fWidth / (GLfloat)fHeight));
		border[1] = -1.0f * ORTHO;
		border[2] = 0.0f;
		border[3] = 1.0f * (ORTHO * ((GLfloat)fWidth / (GLfloat)fHeight));
		border[4] = -1.0f * ORTHO;
		border[5] = 0.0f;
		border[6] = 1.0f * (ORTHO * ((GLfloat)fWidth / (GLfloat)fHeight));
		border[7] = 1.0f * ORTHO;
		border[8] = 0.0f;
		border[9] = -1.0f * (ORTHO * ((GLfloat)fWidth / (GLfloat)fHeight));
		border[10] = 1.0f * ORTHO;
		border[11] = 0.0f;
		border[12] = -1.0f * (ORTHO * ((GLfloat)fWidth / (GLfloat)fHeight));
		border[13] = -1.0f * ORTHO;
		border[14] = 0.0f;
	}
	else
	{
		border[0] = -1.0f * ORTHO;
		border[1] = -1.0f * (ORTHO * ((GLfloat)fHeight / (GLfloat)fWidth));
		border[2] = 0.0f;
		border[3] = 1.0f * ORTHO;
		border[4] = -1.0f * (ORTHO * ((GLfloat)fHeight / (GLfloat)fWidth));
		border[5] = 0.0f;
		border[6] = 1.0f * ORTHO;
		border[7] = 1.0f * (ORTHO * ((GLfloat)fHeight / (GLfloat)fWidth));
		border[8] = 0.0f;
		border[9] = -1.0f * ORTHO;
		border[10] = 1.0f * (ORTHO * ((GLfloat)fHeight / (GLfloat)fWidth));
		border[11] = 0.0f;
		border[12] = -1.0f * ORTHO;
		border[13] = -1.0f * (ORTHO * ((GLfloat)fHeight / (GLfloat)fWidth));
		border[14] = 0.0f;
	}

	glUseProgram(BasicColorShader.ShaderObject.uiShaderProgramObject);

	vmath::mat4 ModelMatrix = vmath::translate(0.0f, 0.0f, 0.0f);
	vmath::mat4 ViewMatrix = vmath::mat4::identity();

	vmath::mat4 ScaleMatrix = vmath::scale(0.99f);
	ModelMatrix = ModelMatrix * ScaleMatrix;

	glUniformMatrix4fv(BasicColorShader.uiModelMatrixUniform, 1, GL_FALSE, ModelMatrix);
	glUniformMatrix4fv(BasicColorShader.uiViewMatrixUniform, 1, GL_FALSE, ViewMatrix);
	glUniformMatrix4fv(BasicColorShader.uiProjectionMatrixUniform, 1, GL_FALSE, gOrthographicProjectionMatrix);

	glBindVertexArray(guiVAO);

	glBindBuffer(GL_ARRAY_BUFFER, guiVBOPos);
	glBufferData(GL_ARRAY_BUFFER, sizeof(border), border, GL_DYNAMIC_DRAW);
	glBindTexture(GL_ARRAY_BUFFER, 0);
	glLineWidth(7.0f);
	glDrawArrays(GL_LINE_STRIP, 0, 5);
	glLineWidth(1.0f);
	glBindVertexArray(0);

	glUseProgram(0);
}

void RenderPlanetOntoABorderedViewport(GLuint uiPlanetTexture, GLfloat fScaleFactor, GLuint uiMoonTexture, GLfloat fScaleFactorMoon, bool bMoon)
{
	static float fFadeOut = 1.0f;

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, gFBOViewport.uiFBO);
	
	DrawNoiseSun(fScaleFactor);

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
	
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	//glClear(GL_COLOR_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	glViewport(650, 600, 600.0f, 350.0f);
	
	glUseProgram(BasicQuadRTTTextureShader.ShaderObject.uiShaderProgramObject);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, gFBOViewport.uiColorRenderBuffer);
	glUniform1i(BasicQuadRTTTextureShader.uiTextureSamplerUniform, 0);

	if (false == gbIsRightMouseButtonPressed)
	{
		glUniform1f(BasicQuadRTTTextureShader.uiFadeoutUniform, fFadeOut);
	}
	else
	{
		if (fFadeOut > 0.0f)
		{
			fFadeOut -= 0.01;
		}
		else
		{
			gbIsFadeOutDone = true;
		}
		glUniform1f(BasicQuadRTTTextureShader.uiFadeoutUniform, fFadeOut);
	}

	glBindVertexArray(guiVAORTT);
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glBindVertexArray(0);

	glUseProgram(0);
	
	if (gbIsFadeOutDone)
	{
		fFadeOut = 1.0f;
		gbIsFadeOutDone = false;
		gbIsAnimationDone = true;
		gbIsPickTransitionDone = false;
		gbIsRightMouseButtonPressed = false;
	}
}

void DrawNoiseSun(GLfloat fScaleFactor)
{
	static float fFadeOut = 1.0f;

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, gFBOViewport.uiFBO);

	glClearColor(0.05f, 0.05f, 0.05f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	DrawBorderedViewport(0, 0, gWindowWidth, gWindowHeight);

	glUseProgram(SunShader.ShaderObject.uiShaderProgramObject);

	vmath::mat4 ScaleMatrix = vmath::mat4::identity();
	vmath::mat4 MVMatrix = vmath::mat4::identity();

	MVMatrix = vmath::translate(0.0f, 0.0f, -10.0f);

	ScaleMatrix = vmath::scale(fScaleFactor);
	MVMatrix = MVMatrix * ScaleMatrix;

	glUniformMatrix4fv(SunShader.uiMVMatrix, 1, GL_FALSE, MVMatrix);
	glUniformMatrix4fv(SunShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

	if (fOffset <= 0.85 && gbIncrease == true)
	{
		fOffset += 0.0005f;
		if (fOffset == 0.85)
		{
			gbIncrease = false;
		}
	}
	else
	{
		gbIncrease = false;
		fOffset -= 0.0005f;
		if (fOffset <= 0.65)
		{
			gbIncrease = true;
		}
	}

	glUniform1f(SunShader.uiOffsetUniform, fOffset);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, noise3DTexName);
	glUniform1i(SunShader.uiTextureSamplerUniform, 0);

	DrawPlanet(&Planet);

	glUseProgram(0);

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	//glClear(GL_COLOR_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	glViewport(350, 300, 600.0f, 350.0f);

	glUseProgram(BasicQuadRTTTextureShader.ShaderObject.uiShaderProgramObject);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, gFBOViewport.uiColorRenderBuffer);
	glUniform1i(BasicQuadRTTTextureShader.uiTextureSamplerUniform, 0);

	if (false == gbIsRightMouseButtonPressed)
	{
		glUniform1f(BasicQuadRTTTextureShader.uiFadeoutUniform, fFadeOut);
	}
	else
	{
		if (fFadeOut > 0.0f)
		{
			fFadeOut -= 0.01;
		}
		else
		{
			gbIsFadeOutDone = true;
		}
		glUniform1f(BasicQuadRTTTextureShader.uiFadeoutUniform, fFadeOut);
	}

	glBindVertexArray(guiVAORTT);
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glBindVertexArray(0);

	glUseProgram(0);

	if (gbIsFadeOutDone)
	{
		fFadeOut = 1.0f;
		gbIsFadeOutDone = false;
		gbIsAnimationDone = true;
		gbIsPickTransitionDone = false;
		gbIsRightMouseButtonPressed = false;
	}
}

void DrawNoiseMarble(GLfloat fScaleFactor, GLfloat fScaleFactorMoon, bool bMoon, vmath::vec3 vecColor1, vmath::vec3 vecColor2)
{
	static float fFadeOut = 1.0f;

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, gFBOViewport.uiFBO);

	glClearColor(0.05f, 0.05f, 0.05f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	DrawBorderedViewport(0, 0, gWindowWidth, gWindowHeight);

	glUseProgram(MarbalShader.ShaderObject.uiShaderProgramObject);

	vmath::mat4 ScaleMatrix = vmath::mat4::identity();
	vmath::mat4 MVMatrix = vmath::mat4::identity();

	MVMatrix = vmath::translate(0.0f, 0.0f, -10.0f);

	ScaleMatrix = vmath::scale(fScaleFactor);
	MVMatrix = MVMatrix * ScaleMatrix;

	glUniformMatrix4fv(MarbalShader.uiMVMatrix, 1, GL_FALSE, MVMatrix);
	glUniformMatrix4fv(MarbalShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

	/*if (fOffset <= 0.85 && gbIncrease == true)
	{
		fOffset += 0.0005f;
		if (fOffset == 0.85)
		{
			gbIncrease = false;
		}
	}
	else
	{
		gbIncrease = false;
		fOffset -= 0.0005f;
		if (fOffset <= 0.65)
		{
			gbIncrease = true;
		}
	}*/

	glUniform1f(MarbalShader.uiOffsetUniform, fOffset);

	glUniform3f(MarbalShader.uiVeinColorUniform, 0.82f, 0.8f, 0.55f);
	glUniform3f(MarbalShader.uiMarbalColorUniform, 0.85f, 0.64f, 0.65f);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, noise3DTexName);
	glUniform1i(MarbalShader.uiTextureSamplerUniform, 0);

	DrawPlanet(&Planet);

	glUseProgram(0);

	if (bMoon)
	{
		// Moon
		vmath::mat4 ModelMatrix = vmath::mat4::identity();
		vmath::mat4 TranslationMatrix = vmath::mat4::identity();
		vmath::mat4 RotationMatrix = vmath::mat4::identity();
		vmath::mat4 ScaleMatrix = vmath::mat4::identity();

		vmath::mat4 ViewMatrix = vmath::lookat(vmath::vec3(0.0f, 0.0f, 10.0f), vmath::vec3(0.0f, 0.0f, 0.0f), vmath::vec3(0.0f, 1.0f, 0.0f));

		glUseProgram(BasicPlanetShader.ShaderObject.uiShaderProgramObject);

		gfMoonTranslationX = (gfSunRadius * fScaleFactor + gfMoonRadius + 0.5f) * (GLfloat)cos(gfMoonAngleTranslate);
		gfMoonTranslationZ = (gfSunRadius * fScaleFactor + gfMoonRadius + 0.5f) * (GLfloat)sin(gfMoonAngleTranslate);

		TranslationMatrix = vmath::mat4::identity();
		TranslationMatrix = vmath::translate(gfMoonTranslationX, 0.0f, -gfMoonTranslationZ);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		RotationMatrix = vmath::rotate(-90.0f, 1.0f, 0.0f, 0.0f);
		RotationMatrix = RotationMatrix * vmath::rotate(gfMoonAngle, 0.0f, 0.0f, 1.0f);
		ModelMatrix = ModelMatrix * RotationMatrix;

		ScaleMatrix = vmath::scale(fScaleFactorMoon);
		ModelMatrix = ModelMatrix * ScaleMatrix;

		glUniformMatrix4fv(BasicPlanetShader.uiModelMatrixUniform, 1, GL_FALSE, ModelMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiViewMatrixUniform, 1, GL_FALSE, ViewMatrix);
		glUniformMatrix4fv(BasicPlanetShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, guiTexturePlanets[PLANET_AND_MOONS::EARTH_MOON]);
		glUniform1i(BasicPlanetShader.uiTextureSamplerUniform, 0);

		DrawPlanet(&SphereMoon);

		glUseProgram(0);
	}

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	//glClear(GL_COLOR_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	glViewport(350, 300, 600.0f, 350.0f);

	glUseProgram(BasicQuadRTTTextureShader.ShaderObject.uiShaderProgramObject);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, gFBOViewport.uiColorRenderBuffer);
	glUniform1i(BasicQuadRTTTextureShader.uiTextureSamplerUniform, 0);

	if (false == gbIsRightMouseButtonPressed)
	{
		glUniform1f(BasicQuadRTTTextureShader.uiFadeoutUniform, fFadeOut);
	}
	else
	{
		if (fFadeOut > 0.0f)
		{
			fFadeOut -= 0.01;
		}
		else
		{
			gbIsFadeOutDone = true;
		}
		glUniform1f(BasicQuadRTTTextureShader.uiFadeoutUniform, fFadeOut);
	}

	glBindVertexArray(guiVAORTT);
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glBindVertexArray(0);

	glUseProgram(0);

	if (gbIsFadeOutDone)
	{
		fFadeOut = 1.0f;
		gbIsFadeOutDone = false;
		gbIsAnimationDone = true;
		gbIsPickTransitionDone = false;
		gbIsRightMouseButtonPressed = false;
	}
}