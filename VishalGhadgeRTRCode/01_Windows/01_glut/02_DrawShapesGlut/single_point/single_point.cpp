#include <GL\freeglut.h>

bool boFullScreen = false;

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

	glutCreateWindow("GLUT- single point");

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
	glClear(GL_COLOR_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glPointSize(10.0f);
	glColor3f(1.0f, 0.0f, 0.0f);

	glBegin(GL_POINTS);
	glVertex3f(0.0f, 0.0f, 0.0f);
	glEnd();

	glutSwapBuffers();
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
