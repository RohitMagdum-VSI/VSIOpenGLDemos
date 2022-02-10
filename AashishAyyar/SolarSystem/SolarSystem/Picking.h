#pragma once

typedef struct _POINT3D 
{
	FLOAT x;
	FLOAT y;
	FLOAT z;
}POINT3D, *P_POINT3D;

typedef struct _PICKING_DATA
{	
	PLANETS_AND_SATELLITES PickedPlanet;
	GLfloat PickedPlanetAngle;	
}PICKING_DATA, *P_PICKING_DATA;

void DrawAllPlanetsPicking();
void GetPickedObject();
BOOL IsObjectPicked();
void DrawPickedObject();
POINT3D GetNextPoint(POINT3D start, POINT3D end, FLOAT z);
void GetEndOffsetForSpecificPlanet(PLANETS_AND_SATELLITES Planet, POINT3D &Point);

