void drawTextConst()
{
	glLineWidth(3.0f);
	GLfloat depth = -10.0f;
    glLoadIdentity();
    glTranslatef(-1.0f, 0.0f, depth);
    Draw_C();

	glLoadIdentity();
    glTranslatef(-0.5f, 0.0f, depth);
    Draw_O();

	

    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, depth);
    Draw_N();

    glLoadIdentity();
    glTranslatef(0.8f, 0.0f, depth);
    Draw_S();
    
    glLoadIdentity();
    glTranslatef(1.4f, 0.0f, depth);
    Draw_T();


    
}
