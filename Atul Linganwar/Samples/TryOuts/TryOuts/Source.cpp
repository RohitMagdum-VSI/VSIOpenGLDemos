// header files
#include<GL/freeglut.h>
#include<cmath>


// global variables
bool bFullScreen = false;

// Entry point function argc - argument count, argv - arguent vector
int main(int argc, char* argv[])
{
	// local function declarations
	void initialize(void);
	void resize(int, int);
	void display(void);
	void keyboard(unsigned char, int, int);
	void mouse(int, int, int, int);
	void uninitialize(void);

	// code
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);

	glutInitWindowSize(800, 600);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("Akash Jeevan Ghule");

	initialize();

	glutReshapeFunc(resize);
	glutDisplayFunc(display);

	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutCloseFunc(uninitialize);

	glutMainLoop();

	return(0);
}

// Functions
void initialize(void)
{
	// code
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}

void resize(int width, int height)
{
	// code
	glViewport(0, 0, width, height);
}

void display(void)
{
	// code

		// code
	glClear(GL_COLOR_BUFFER_BIT);

	// glBegin(GL_LINES);

		// glColor3f(1.0f, 1.0f, 1.0f);
		// glPointSize(10.0f);


		// // Draw circle
		// double PI = 3.1415926535898;
		// GLint circle_points = 50;
		// int i;
		// float angle;
		// for (i = 0; i < circle_points ; i++) 
		// {
			// angle = 2*PI*i/circle_points; 
			// glVertex3f (0.0f, cos (angle), sin (angle) );
		// }

	// glEnd();
	// glutSwapBuffers();


	glBegin(GL_LINE_LOOP);

	glColor3f(1.0f, 1.0f, 1.0f);
	glPointSize(10.0f);
	//glVertex3f(0.0f, 0.0f, 0.0f); // Center

	// Draw circle
	double PI = 3.1415926535898;
	GLint circle_points = 100;
	int i;
	float angle;
	for (i = 0; i < circle_points; i++)
	{
		angle = 2 * PI * i / circle_points;
		glVertex2f(cos(angle), sin(angle));
	}


	glEnd();
	//glutSwapBuffers();


	// 12 lines
	glBegin(GL_LINE_LOOP);
	{
		glColor3f(1.0f, 0.0f, 0.0f);
		glPointSize(10.0f);
		float quadrant_v1 = 0.45f, quadrant_v2 = 0.81f, axis_v1 = 0.9f; //, axis_y = 0.5f;

		glVertex3f(0.0f, 0.0f, 0.0f); // Center

		glVertex3f(-axis_v1, 0.0f, 0.0f); // Left of window
		glVertex3f(0.0f, 0.0f, 0.0f); // Center

		glVertex3f(-quadrant_v2, quadrant_v1, 0.0f); // 2nd quadrant
		glVertex3f(0.0f, 0.0f, 0.0f); // Center

		glVertex3f(-quadrant_v1, quadrant_v2, 0.0f); // 2nd quadrant
		glVertex3f(0.0f, 0.0f, 0.0f); // Center

		glVertex3f(0.0f, axis_v1, 0.0f); // Top of window
		glVertex3f(0.0f, 0.0f, 0.0f); // Center

		glVertex3f(quadrant_v1, quadrant_v2, 0.0f); // 1st quadrant
		glVertex3f(0.0f, 0.0f, 0.0f); // Center

		glVertex3f(quadrant_v2, quadrant_v1, 0.0f); // 1st quadrant
		glVertex3f(0.0f, 0.0f, 0.0f); // Center

		glVertex3f(axis_v1, 0.0f, 0.0f); // Right of window
		glVertex3f(0.0f, 0.0f, 0.0f); // Center


		glVertex3f(quadrant_v2, -quadrant_v1, 0.0f); // 4th quadrant
		glVertex3f(0.0f, 0.0f, 0.0f); // Center

		glVertex3f(quadrant_v1, -quadrant_v2, 0.0f); // 4th quadrant
		glVertex3f(0.0f, 0.0f, 0.0f); // Center


		glVertex3f(0.0f, -axis_v1, 0.0f); // Bottom of window
		glVertex3f(0.0f, 0.0f, 0.0f); // Center


		glColor3f(1.0f, 0.0f, 0.0f);
		glVertex3f(0.0f, 0.0f, 0.0f); // Center

		glVertex3f(-quadrant_v1, -quadrant_v2, 0.0f); // 3rd quadrant
		glVertex3f(0.0f, 0.0f, 0.0f); // Center

		glVertex3f(-quadrant_v2, -quadrant_v1, 0.0f); // 3rd quadrant
		glVertex3f(0.0f, 0.0f, 0.0f); // Center
	}
	glEnd();
	glutSwapBuffers();

}

void keyboard(unsigned char key, int x, int y)
{
	// code
	switch (key)
	{
	case 27:
		glutLeaveMainLoop();
		break;

	case 'F':
	case 'f':
		if (bFullScreen == false)
		{
			glutFullScreen();
			bFullScreen = true;
		}
		else
		{
			glutLeaveFullScreen();
			bFullScreen = false;
		}
		break;

	default:
		break;
	}
}

void mouse(int button, int state, int x, int y)
{
	// code
	switch (button)
	{
	case GLUT_LEFT_BUTTON:
		glutLeaveMainLoop();
		break;

	default:
		break;
	}
}

void uninitialize()
{
	// code
}
