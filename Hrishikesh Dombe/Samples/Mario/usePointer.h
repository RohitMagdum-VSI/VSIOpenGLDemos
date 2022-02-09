void drawTextUsePointer()
{
	GLfloat depth = -10.0f;
	glLineWidth(3.0f);
	glLoadIdentity();
	glTranslatef(-4.0f, 0.0f, depth);
	Draw_U();

	glLoadIdentity();
	glTranslatef(-3.2f, 0.0f, depth);
	Draw_S();

	glLoadIdentity();
	glTranslatef(-2.5f, 0.0f, depth);
	Draw_E();

	glLoadIdentity();
	glTranslatef(-1.25f, 0.0f, depth);
	Draw_P();

	glLoadIdentity();
	glTranslatef(-0.35f, 0.0f, depth);
	Draw_O();

	glLoadIdentity();
	glTranslatef(0.0f, 0.0f, depth);
	Draw_I();

	glLoadIdentity();
	glTranslatef(0.75f, 0.0f, depth);
	Draw_N();

	glLoadIdentity();
	glTranslatef(1.5f, 0.0f, depth);
	Draw_T();

	glLoadIdentity();
	glTranslatef(2.25f, 0.0f, depth);
	Draw_E();

	glLoadIdentity();
	glTranslatef(3.0f, 0.0f, depth);
	Draw_R();
}