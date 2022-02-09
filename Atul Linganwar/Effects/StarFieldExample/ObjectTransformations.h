#pragma once

typedef struct _TRANSLATE
{
	GLfloat fX;
	GLfloat fY;
	GLfloat fZ;

}TRANSLATE, *PTRANSLATE;

typedef struct _ROTATE
{
	GLfloat fAngle;
	GLfloat fX;
	GLfloat fY;
	GLfloat fZ;

}ROTATE, *PROTATE;

typedef struct _SCALE
{
	GLfloat fX;
	GLfloat fY;
	GLfloat fZ;

}SCALE, *PSCALE;

typedef struct _OBJECT_TRANSFORMATION
{
	TRANSLATE translation;
	ROTATE rotation;
	SCALE scale;

}OBJECT_TRANSFORMATION, *POBJECT_TRANSFORMATION;

