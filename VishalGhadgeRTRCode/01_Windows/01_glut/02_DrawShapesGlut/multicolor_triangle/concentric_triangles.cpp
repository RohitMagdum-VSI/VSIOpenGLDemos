#include <GL\freeglut.h>

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

	glutCreateWindow("GLUT- Concentric Triangles");

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
	VOID DrawConcentricTriangles();

	glClear(GL_COLOR_BUFFER_BIT);

	DrawConcentricTriangles();

	glutSwapBuffers();
}


VOID DrawConcentricTriangles()
{
	int i = 0;

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	//
	//	Draw triangle
	//
	glLineWidth(1.0f);
	glColor3f(1.0f, 1.0f, 0.0f);
	for (GLfloat glfTemp = 0.1f; glfTemp <= 1.1f && i < 10; glfTemp += 0.1f, i++)
	{
		glBegin(GL_LINE_STRIP);
		glColor3f(g_glfarrColor[i][0], g_glfarrColor[i][1], g_glfarrColor[i][2]);
		glVertex3f(0.0f, glfTemp, 0.0f);
		glVertex3f(-glfTemp, -glfTemp, 0.0f);
		glVertex3f(glfTemp, -glfTemp, 0.0f);
		glVertex3f(0.0f, glfTemp, 0.0f);
		glEnd();
	}
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
