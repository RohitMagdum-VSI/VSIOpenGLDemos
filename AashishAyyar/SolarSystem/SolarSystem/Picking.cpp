#include "Common.h"

#define OBJECT_IN_FOCUS_X 0.0f
#define OBJECT_IN_FOCUS_Y 19000.0f
#define OBJECT_IN_FOCUS_Z 24000.0f

#define EXP_INCREMENT 2.0f;

extern PLANET gPlanet;
extern ORBIT gOrbit;
extern PICKING_SHADER gPickingShader;
extern mat4 gPerspectiveProjectionMatrix;
extern mat4 gViewMatrix;
extern FRAME_BUFFER gFrameBuffer;

extern FLOAT gLeftMouseButtonX;
extern FLOAT gLeftMouseButtonY;

extern PICKING_DATA gPickedObjectData;

extern FLOAT gfPlanetAngle;

extern GLuint gPlanetTextures[10];

extern SATURN_RING gSaturnRing;

FLOAT fPickedObjectZTranslate = 0.0f;
FLOAT exponential = 0.0f;

void DrawPlanetPicking(PLANET &Planet, PICKING_SHADER &PickingShader, vmath::mat4 modelMatrix, UINT objectID)
{
	glUseProgram(PickingShader.Shader.renderProgram);

	glUniformMatrix4fv(PickingShader.modelMatrixUniform, 1, GL_FALSE, modelMatrix);
	glUniformMatrix4fv(PickingShader.viewMatrixUniform, 1, GL_FALSE, gViewMatrix);
	glUniformMatrix4fv(PickingShader.projectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);
	glUniform1i(PickingShader.objectIDUniform, (GLint)objectID);

	DrawSphere(Planet.Sphere);

	glUseProgram(0);
}

void DrawRingPicking(RING &Ring, PICKING_SHADER &PickingShader, vmath::mat4 modelMatrix, UINT objectID)
{
	glUseProgram(PickingShader.Shader.renderProgram);

	glUniformMatrix4fv(PickingShader.modelMatrixUniform, 1, GL_FALSE, modelMatrix);
	glUniformMatrix4fv(PickingShader.viewMatrixUniform, 1, GL_FALSE, gViewMatrix);
	glUniformMatrix4fv(PickingShader.projectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);
	glUniform1i(PickingShader.objectIDUniform, (GLint)objectID);

	DrawRing(Ring);

	glUseProgram(0);
}


void DrawTransformedPlanetPicking(FLOAT xPos, FLOAT zPos, FLOAT fPlanetScale, UINT uiObjectID, BOOL bRotate = TRUE)
{
	mat4 translationMatrix = mat4::identity();
	mat4 rotationMatrix = mat4::identity();
	mat4 scaleMatrix = mat4::identity();
	mat4 modelMatrix = mat4::identity();

	static float fAngle = 0.0f;
	translationMatrix = translate(xPos, 0.0f, zPos);
	rotationMatrix *= rotate(270.0f, 1.0f, 0.0f, 0.0f);

	if (bRotate)
		rotationMatrix *= rotate(fAngle, 0.0f, 0.0f, 1.0f);

	scaleMatrix = scale(fPlanetScale, fPlanetScale, fPlanetScale);

	modelMatrix = translationMatrix * rotationMatrix * scaleMatrix;
	DrawPlanetPicking(gPlanet, gPickingShader, modelMatrix, uiObjectID);

	fAngle += 0.1f;
}

void DrawTransformedRingPicking(FLOAT xPos, FLOAT zPos, FLOAT fPlanetScale, UINT uiObjectID, BOOL bRotate = TRUE)
{
	mat4 translationMatrix = mat4::identity();
	mat4 rotationMatrix = mat4::identity();
	mat4 scaleMatrix = mat4::identity();
	mat4 modelMatrix = mat4::identity();

	static float fAngle = 0.0f;
	translationMatrix = translate(xPos, 0.0f, zPos);
	rotationMatrix *= rotate(270.0f, 1.0f, 0.0f, 0.0f);

	if (bRotate)
		rotationMatrix *= rotate(fAngle, 0.0f, 0.0f, 1.0f);

	scaleMatrix = scale(fPlanetScale, fPlanetScale, fPlanetScale);

	modelMatrix = translationMatrix * rotationMatrix * scaleMatrix;
	DrawRingPicking(gSaturnRing.Ring, gPickingShader, modelMatrix, uiObjectID);

	fAngle += 0.1f;
}

void DrawAllPlanetsPicking()
{
	//
	//	Sun
	//
	DrawTransformedPlanetPicking(
		0.0f,
		0.0f,
		GetPlanetScale(PLANETS_AND_SATELLITES::SUN),
		PLANETS_AND_SATELLITES::SUN,
		FALSE);

	//
	//	Mercury
	//
	DrawTransformedPlanetPicking(
		GetPlanetXPosition(gOrbit, PLANETS_AND_SATELLITES::MERCURY, ANGLE_WITH_OFFSET(gfPlanetAngle, MERCURY_OFFSET)),
		GetPlanetZPosition(gOrbit, PLANETS_AND_SATELLITES::MERCURY, ANGLE_WITH_OFFSET(gfPlanetAngle, MERCURY_OFFSET)),
		GetPlanetScale(PLANETS_AND_SATELLITES::MERCURY),
		PLANETS_AND_SATELLITES::MERCURY
		);

	//
	//	Venus
	//
	DrawTransformedPlanetPicking(
		GetPlanetXPosition(gOrbit, PLANETS_AND_SATELLITES::VENUS, ANGLE_WITH_OFFSET(gfPlanetAngle, VENUS_OFFSET)),
		GetPlanetZPosition(gOrbit, PLANETS_AND_SATELLITES::VENUS, ANGLE_WITH_OFFSET(gfPlanetAngle, VENUS_OFFSET)),
		GetPlanetScale(PLANETS_AND_SATELLITES::VENUS),
		PLANETS_AND_SATELLITES::VENUS
	);

	//
	//	Earth
	//
	DrawTransformedPlanetPicking(
		GetPlanetXPosition(gOrbit, PLANETS_AND_SATELLITES::EARTH, ANGLE_WITH_OFFSET(gfPlanetAngle, EARTH_OFFSET)),
		GetPlanetZPosition(gOrbit, PLANETS_AND_SATELLITES::EARTH, ANGLE_WITH_OFFSET(gfPlanetAngle, EARTH_OFFSET)),
		GetPlanetScale(PLANETS_AND_SATELLITES::EARTH),
		PLANETS_AND_SATELLITES::EARTH
	);

	//
	//	Mars
	//
	DrawTransformedPlanetPicking(
		GetPlanetXPosition(gOrbit, PLANETS_AND_SATELLITES::MARS, ANGLE_WITH_OFFSET(gfPlanetAngle, MARS_OFFSET)),
		GetPlanetZPosition(gOrbit, PLANETS_AND_SATELLITES::MARS, ANGLE_WITH_OFFSET(gfPlanetAngle, MARS_OFFSET)),
		GetPlanetScale(PLANETS_AND_SATELLITES::MARS),
		PLANETS_AND_SATELLITES::MARS
	);

	//
	//	Jupiter
	//
	DrawTransformedPlanetPicking(
		GetPlanetXPosition(gOrbit, PLANETS_AND_SATELLITES::JUPITER, ANGLE_WITH_OFFSET(gfPlanetAngle, JUPITER_OFFSET)),
		GetPlanetZPosition(gOrbit, PLANETS_AND_SATELLITES::JUPITER, ANGLE_WITH_OFFSET(gfPlanetAngle, JUPITER_OFFSET)),
		GetPlanetScale(PLANETS_AND_SATELLITES::JUPITER),
		PLANETS_AND_SATELLITES::JUPITER
	);

	//
	//	Saturn
	//

	DrawTransformedRingPicking(
		GetPlanetXPosition(gOrbit, PLANETS_AND_SATELLITES::SATURN, ANGLE_WITH_OFFSET(gfPlanetAngle, SATURN_OFFSET)),
		GetPlanetZPosition(gOrbit, PLANETS_AND_SATELLITES::SATURN, ANGLE_WITH_OFFSET(gfPlanetAngle, SATURN_OFFSET)),
		1.0f,
		PLANETS_AND_SATELLITES::SATURN_RING_ID);

	DrawTransformedPlanetPicking(
		GetPlanetXPosition(gOrbit, PLANETS_AND_SATELLITES::SATURN, ANGLE_WITH_OFFSET(gfPlanetAngle, SATURN_OFFSET)),
		GetPlanetZPosition(gOrbit, PLANETS_AND_SATELLITES::SATURN, ANGLE_WITH_OFFSET(gfPlanetAngle, SATURN_OFFSET)),
		GetPlanetScale(PLANETS_AND_SATELLITES::SATURN),
		PLANETS_AND_SATELLITES::SATURN
	);

	//
	//	Uranus
	//
	DrawTransformedPlanetPicking(
		GetPlanetXPosition(gOrbit, PLANETS_AND_SATELLITES::URANUS, ANGLE_WITH_OFFSET(gfPlanetAngle, URANUS_OFFSET)),
		GetPlanetZPosition(gOrbit, PLANETS_AND_SATELLITES::URANUS, ANGLE_WITH_OFFSET(gfPlanetAngle, URANUS_OFFSET)),
		GetPlanetScale(PLANETS_AND_SATELLITES::URANUS),
		PLANETS_AND_SATELLITES::URANUS
	);

	//
	//	Neptune
	//
	DrawTransformedPlanetPicking(
		GetPlanetXPosition(gOrbit, PLANETS_AND_SATELLITES::NEPTUNE, ANGLE_WITH_OFFSET(gfPlanetAngle, NEPTUNE_OFFSET)),
		GetPlanetZPosition(gOrbit, PLANETS_AND_SATELLITES::NEPTUNE, ANGLE_WITH_OFFSET(gfPlanetAngle, NEPTUNE_OFFSET)),
		GetPlanetScale(PLANETS_AND_SATELLITES::NEPTUNE),
		PLANETS_AND_SATELLITES::NEPTUNE
	);

	//
	//	Pluto
	//
	DrawTransformedPlanetPicking(
		GetPlanetXPosition(gOrbit, PLANETS_AND_SATELLITES::PLUTO, ANGLE_WITH_OFFSET(gfPlanetAngle, PLUTO_OFFSET)),
		GetPlanetZPosition(gOrbit, PLANETS_AND_SATELLITES::PLUTO, ANGLE_WITH_OFFSET(gfPlanetAngle, PLUTO_OFFSET)),
		GetPlanetScale(PLANETS_AND_SATELLITES::PLUTO),
		PLANETS_AND_SATELLITES::PLUTO
	);

	return;
}

UCHAR GetPickedFragmentData()
{
	GLint viewPort[4] = { 0 };
	UCHAR data[4] = { 0 };
		
	glGetIntegerv(GL_VIEWPORT, viewPort);
	
	glBindFramebuffer(GL_READ_FRAMEBUFFER, gFrameBuffer.fbo);
		glReadBuffer(GL_COLOR_ATTACHMENT0);
		glReadPixels(gLeftMouseButtonX, viewPort[3] - gLeftMouseButtonY, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, &data);
		glReadBuffer(GL_NONE);
	glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

	return data[0];
}

void GetPickedObject()
{
	PLANETS_AND_SATELLITES Enum = (PLANETS_AND_SATELLITES)GetPickedFragmentData();

	if (Enum != PLANETS_AND_SATELLITES::NONE)
	{
		if (Enum == PLANETS_AND_SATELLITES::SATURN_RING_ID)
			Enum = PLANETS_AND_SATELLITES::SATURN;

		gPickedObjectData.PickedPlanet = Enum;
		gPickedObjectData.PickedPlanetAngle = gfPlanetAngle;

		fPickedObjectZTranslate = 0;
		exponential = 0;

		/*if (gPickedObjectData.PickedPlanet == PLANETS_AND_SATELLITES::SUN)
			PlaySound(L"C:\\Users\\ayyar\\Music\\ChadtaSuraj.wav", NULL, SND_FILENAME | SND_ASYNC);*/
	}

	return;
}

BOOL IsObjectPicked() 
{
	if (gPickedObjectData.PickedPlanet == PLANETS_AND_SATELLITES::NONE)
		return FALSE;
	else 
		return TRUE;
}

void DrawPickedObject() 
{
	FLOAT x = GetPlanetXPosition(gOrbit, gPickedObjectData.PickedPlanet, ANGLE_WITH_OFFSET(gPickedObjectData.PickedPlanetAngle, GetPlanetOffset(gPickedObjectData.PickedPlanet)));
	FLOAT z = GetPlanetZPosition(gOrbit, gPickedObjectData.PickedPlanet, ANGLE_WITH_OFFSET(gPickedObjectData.PickedPlanetAngle, GetPlanetOffset(gPickedObjectData.PickedPlanet)));

	POINT3D start = { x, 0.0f, z};
	POINT3D end = { OBJECT_IN_FOCUS_X, OBJECT_IN_FOCUS_Y, OBJECT_IN_FOCUS_Z };

	//GetEndOffsetForSpecificPlanet(gPickedObjectData.PickedPlanet, end);

	POINT3D currentPosition = GetNextPoint(start, end, z + fPickedObjectZTranslate);

	if (gPickedObjectData.PickedPlanet == PLANETS_AND_SATELLITES::SATURN)
	{
		DrawTransformedRing(
			currentPosition.x,
			currentPosition.y,
			currentPosition.z,
			1.0f,
			gPlanetTextures[PLANETS_AND_SATELLITES::SATURN_RING_ID],
			TRUE
		);
	}

	DrawTransformedPlanet(
		currentPosition.x,
		currentPosition.y,
		currentPosition.z,
		GetPlanetScale(gPickedObjectData.PickedPlanet),
		gPlanetTextures[gPickedObjectData.PickedPlanet]
	);

	if (currentPosition.z < OBJECT_IN_FOCUS_Z)
	{
		fPickedObjectZTranslate = fPickedObjectZTranslate + exponential;
		exponential = exponential + EXP_INCREMENT;
	}


}

POINT3D GetNextPoint(POINT3D start, POINT3D end, FLOAT z) 
{
	POINT3D p = { 0 };

	p.x = (((end.x - start.x) / (end.z - start.z)) * (z - start.z)) + start.x;
	p.y = (((end.y - start.y) / (end.z - start.z)) * (z - start.z)) + start.y;
	p.z = z;
	
	return p;
}

BOOL IsMoonPresent(PLANETS_AND_SATELLITES Planet) 
{
	if (
		(Planet == PLANETS_AND_SATELLITES::EARTH) ||
		(Planet == PLANETS_AND_SATELLITES::MARS) ||
		(Planet == PLANETS_AND_SATELLITES::SATURN) ||
		(Planet == PLANETS_AND_SATELLITES::JUPITER) ||
		(Planet == PLANETS_AND_SATELLITES::NEPTUNE) ||
		(Planet == PLANETS_AND_SATELLITES::URANUS) ||
		(Planet == PLANETS_AND_SATELLITES::PLUTO)
	)
	{
		return TRUE;
	}
	
	return FALSE;
}

void GetEndOffsetForSpecificPlanet(PLANETS_AND_SATELLITES Planet, POINT3D &Point) 
{
	switch (Planet) 
	{
	//
	//	LARGE PLANETS
	//
	case SUN:
		Point.y -= 1000.0f;
		Point.z -= 1000.0f;
		break;
	
	//
	//	MEDIUM SIZED PLANETS
	//
	case EARTH:
		Point.x -= 2000.0f;
		Point.y += 4000.0f;
		Point.z += 5000.0f;
		break;

	//
	//	SMALL SIZED PLANETS
	//
	case MERCURY:
		Point.x -= 1750.0f;
		Point.y += 12000.0f;
		Point.z += 15000.0f;
		break;	
	}
}