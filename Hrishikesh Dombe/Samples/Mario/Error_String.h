
GLfloat charTranslateX = 0.0f;
GLfloat charTranslateY = 0.0f;
GLfloat charTranslateZ = 0.0f;


/*void resize(int width, int height)
{
	if (height == 0)
		height = 1;
	glViewport(0, 0, (GLsizei)width, (GLsizei)height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0f, (GLdouble)(width / height), 1.0f, 100.0f);
}*/


void DrawAddressingModes(void)
{
	charTranslateX = -2.7f;
	charTranslateY = 0.7f;
	charTranslateZ = -10.0f;
	glLoadIdentity();
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_A();
	
	glLoadIdentity();
	charTranslateX = charTranslateX + 0.4f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_D();
	
	glLoadIdentity();
	charTranslateX = charTranslateX + 0.70f;
	glTranslatef(charTranslateX , charTranslateY, charTranslateZ);
	Draw_D();


	glLoadIdentity();
	charTranslateX = charTranslateX + 0.70f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_R();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.55f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_E();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.60f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_S();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.60f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_S();


	glLoadIdentity();
	charTranslateX = charTranslateX + 0.55f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_I();


	glLoadIdentity();
	charTranslateX = charTranslateX + 0.62f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_N();

	glLoadIdentity();
	charTranslateX = charTranslateX + 1.00f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_G();

	charTranslateX = -1.5f;
	charTranslateY = -0.7f;
	charTranslateZ = -9.0f;

	glLoadIdentity();
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_M();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.90f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_O();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.4f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_D();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.7f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_E();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.7f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_S();
}

void DrawStackManipulation(void)
{
	charTranslateX = -1.3f;
	charTranslateY = 0.7f;
	charTranslateZ = -10.0f;
	
	glLoadIdentity();
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_S();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.55f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_T();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.82f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_A();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.85f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_C();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.2f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_K();

	charTranslateX = -3.7f;
	charTranslateY = -0.7f;
	charTranslateZ = -11.0f;

	glLoadIdentity();
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_M();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.90f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_A();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.4f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_N();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.65f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_I();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.60f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_P();


	glLoadIdentity();
	charTranslateX = charTranslateX + 0.60f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_U();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.60f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_L();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.85f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_A();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.4f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_T();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.65f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_I();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.75f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_O();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.45f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_N();
}

void DrawFloatingPointComputation(void)
{
	charTranslateX = -2.2f;
	charTranslateY = 1.7f;
	charTranslateZ = -10.0f;

	glLoadIdentity();
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_F();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.55f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_L();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.82f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_O();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.60f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_A();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.4f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_T();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.7f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_I();
	
	glLoadIdentity();
	charTranslateX = charTranslateX + 0.7f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_N();

	glLoadIdentity();
	charTranslateX = charTranslateX + 1.1f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_G();

	charTranslateX = -1.40f;
	charTranslateY = 0.0f;
	charTranslateZ = -10.0f;

	glLoadIdentity();
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_P();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.85f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_O();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.4f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_I();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.65f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_N();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.60f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_T();

	charTranslateX = -2.7f;
	charTranslateY = -1.7f;
	charTranslateZ = -10.0f;

	glLoadIdentity();
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_C();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.47f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_O();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.4f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_M();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.65f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_P();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.60f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_U();


	glLoadIdentity();
	charTranslateX = charTranslateX + 0.60f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_T();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.85f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_A();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.4f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_T();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.65f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_I();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.75f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_O();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.45f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_N();

}

void DrawISAParallelism(void)
{
	charTranslateX = -1.2f;
	charTranslateY = 0.7f;
	charTranslateZ = -10.0f;

	glLoadIdentity();
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_I();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.65f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_S();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.92f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_A();

	charTranslateX = -3.7f;
	charTranslateY = -0.7f;
	charTranslateZ = -11.0f;

	glLoadIdentity();
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_P();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.80f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_A();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.5f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_R();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.75f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_A();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.40f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_L();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.60f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_L();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.60f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_E();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.70f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_S();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.55f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_I();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.75f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_O();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.45f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_N();
}

void DrawSegmentationFault(void)
{
	charTranslateX = -4.0f;
	charTranslateY = 0.7f;
	charTranslateZ = -10.0f;

	glLoadIdentity();
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_S();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.65f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_E();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.95f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_G();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.50f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_M();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.67f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_E();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.63f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_N();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.70f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_T();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.80f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_A();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.30f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_T();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.70f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_I();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.90f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_O();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.53f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_N();

	charTranslateX = -1.5f;
	charTranslateY = -0.7f;
	charTranslateZ = -11.0f;

	glLoadIdentity();
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_F();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.73f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_A();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.38f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_U();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.7f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_L();

	glLoadIdentity();
	charTranslateX = charTranslateX + 0.55f;
	glTranslatef(charTranslateX, charTranslateY, charTranslateZ);
	Draw_T();
}
