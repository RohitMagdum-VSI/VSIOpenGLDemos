#include <GL\freeglut.h>
#include <math.h>

bool boFullScreen = false;

GLfloat g_glfarrColor[10][3] =
{
	{ 1.0f, 0.0f, 0.0f },
	{ 0.0f, 1.0f, 0.0f },
	{ 0.0f, 0.0f, 1.0f },
	{ 0.0f, 1.0f, 1.0f },
	{ 1.0f, 0.0f, 1.0f },
	{ 1.0f, 1.0f, 0.0f },
	{ 0.5f, 0.5f, 0.0f },
	{ 0.0f, 0.5f, 0.5f },
	{ 0.5f, 0.0f, 1.0f },
	{ 0.5f, 0.5f, 0.5f },
};

int main(int argc, char** argv)
{
	//
	//	Function prototype.
	//
	void initialize();
	void display();
	void keyboard(unsigned char key, int x, int y);
	void mouse(int button, int state, int x, int y);
	void resize(int, int);
	void uninitialize();

	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);

	glutInitWindowSize(800, 600);
	glutInitWindowPosition(250, 50);

	glutCreateWindow("GLUT- Concentric Circles");

	initialize();

	glutDisplayFunc(display);
	glutReshapeFunc(resize);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutCloseFunc(uninitialize);

	glutMainLoop();

	//	return(0);
}

void initialize()
{
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
}

void display()
{
	VOID DrawConcentricCircle();

	glClear(GL_COLOR_BUFFER_BIT);

	DrawConcentricCircle();

	glutSwapBuffers();
}


VOID DrawConcentricCircle()
{
	GLfloat glfTemp;
	const float PI = 3.14f;

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glLineWidth(1.0f);
	//
	//	Circle.
	//
	glBegin(GL_POINTS);
	for (GLfloat glfAngle = 0.0f; glfAngle < 2.0f * PI; glfAngle += 0.001f)
	{
		glfTemp = 0.0f;
		glColor3f(1.0f, 0.0f, 0.0f);
		glVertex3f(cos(glfAngle), sin(glfAngle), 0.0f);

		glfTemp = glfTemp + 0.1f;
		glColor3f(0.0f, 1.0f, 0.0f);
		glVertex3f(cos(glfAngle) * glfTemp, sin(glfAngle) * glfTemp, 0.0f);

		glfTemp = glfTemp + 0.1f;
		glColor3f(0.0f, 0.0f, 1.0f);
		glVertex3f(cos(glfAngle) * glfTemp, sin(glfAngle) * glfTemp, 0.0f);

		glfTemp = glfTemp + 0.1f;
		glColor3f(0.0f, 1.0f, 1.0f);
		glVertex3f(cos(glfAngle) * glfTemp, sin(glfAngle) * glfTemp, 0.0f);

		glfTemp = glfTemp + 0.1f;
		glColor3f(1.0f, 0.0f, 1.0f);
		glVertex3f(cos(glfAngle) * glfTemp, sin(glfAngle) * glfTemp, 0.0f);

		glfTemp = glfTemp + 0.1f;
		glColor3f(1.0f, 1.0f, 0.0f);
		glVertex3f(cos(glfAngle) * glfTemp, sin(glfAngle) * glfTemp, 0.0f);

		glfTemp = glfTemp + 0.1f;
		glColor3f(0.5f, 0.5f, 0.0f);
		glVertex3f(cos(glfAngle) * glfTemp, sin(glfAngle) * glfTemp, 0.0f);

		glfTemp = glfTemp + 0.1f;
		glColor3f(0.0f, 0.5f, 0.5f);
		glVertex3f(cos(glfAngle) * glfTemp, sin(glfAngle) * glfTemp, 0.0f);

		glfTemp = glfTemp + 0.1f;
		glColor3f(0.5f, 0.0f, 1.0f);
		glVertex3f(cos(glfAngle) * glfTemp, sin(glfAngle) * glfTemp, 0.0f);

		glfTemp = glfTemp + 0.1f;
		glColor3f(0.5f, 0.5f, 0.5f);
		glVertex3f(cos(glfAngle) * glfTemp, sin(glfAngle) * glfTemp, 0.0f);
	}
	glEnd();
}

void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 27:	//	Escape
		glutLeaveMainLoop();
		break;

	case 'f':
	case 'F':
		if (false == boFullScreen)
		{
			glutFullScreen();
			boFullScreen = true;
		}
		else
		{
			glutLeaveFullScreen();
			boFullScreen = false;
		}
		break;

	default:
		break;
	}
}

void mouse(int button, int state, int x, int y)
{
	switch (button)
	{
	case GLUT_LEFT_BUTTON:
		break;

	default:
		break;
	}
}

void resize(int width, int height)
{
	if (0 == height)
	{
		height = 1;
	}

	glViewport(0, 0, width, height);
}

void uninitialize()
{

}
