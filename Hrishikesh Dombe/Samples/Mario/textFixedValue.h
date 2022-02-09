void drawTextFixedValue()
{
	GLfloat depth = -10.0f;
	glLineWidth(3.0f);
    glLoadIdentity();
    glTranslatef(-3.2f, 0.0f, depth);
    Draw_F();

	glLoadIdentity();
    glTranslatef(-2.6f, 0.0f, depth);
    Draw_I();

	

    glLoadIdentity();
    glTranslatef(-1.9f, 0.0f, depth);
    Draw_X();

    glLoadIdentity();
    glTranslatef(-1.2f, 0.0f, depth);
    Draw_E();
    
    glLoadIdentity();
    glTranslatef(-0.5f, 0.0f, depth);
    Draw_D();


	    glLoadIdentity();
    glTranslatef(0.5f, 0.0f, depth);
    Draw_V();

		    glLoadIdentity();
    glTranslatef(1.4f, 0.0f, depth);
    Draw_A();


	    glLoadIdentity();
    glTranslatef(1.9f, 0.0f, depth);
    Draw_L();


	    glLoadIdentity();
    glTranslatef(2.5f, 0.0f, depth);
    Draw_U();


	    glLoadIdentity();
    glTranslatef(3.2f, 0.0f, depth);
    Draw_E();




}
