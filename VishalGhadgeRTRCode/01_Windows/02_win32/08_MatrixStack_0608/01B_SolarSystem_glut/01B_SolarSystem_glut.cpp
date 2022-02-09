#include <GL\freeglut.h>

bool boFullScreen = false;
static int g_s_iYear;
static int g_s_iDay;


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

	glutCreateWindow("3D : Solar System Glut");

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
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	//	Change 3 for 3D

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	//
	//	View transaformation.
	//
	gluLookAt(0.0f, 0.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);

	glPushMatrix();

	glColor3f(1.0f, 1.0f, 0.0f);
	glutWireSphere(1.0, 30, 30);
	glRotatef((GLfloat)g_s_iYear, 0.0f, 1.0f, 0.0f);
	glTranslatef(2.0f, 0.0f, 0.0f);

	glRotatef((GLfloat)g_s_iDay, 0.0f, 1.0f, 0.0f);

	glColor3f(0.4f, 0.9f, 1.0f);
	glutWireSphere(0.2, 30, 30);
	glPopMatrix();

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

	case 'd':
		g_s_iDay = (g_s_iDay + 6) % 360;
		glutPostRedisplay();
		break;

	case 'D':
		g_s_iDay = (g_s_iDay - 6) % 360;
		glutPostRedisplay();
		break;

	case 'y':
		g_s_iYear = (g_s_iYear + 3) % 360;
		glutPostRedisplay();
		break;

	case 'Y':
		g_s_iYear = (g_s_iYear - 3) % 360;
		glutPostRedisplay();
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

void resize(int iWidth, int iHeight)
{
	if (0 == iHeight)
	{
		iHeight = 1;
	}

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	//
	//	znear and zfar must positive.
	//
	if (iWidth <= iHeight)
	{
		gluPerspective(45, (GLfloat)iHeight / (GLfloat)iWidth, 0.1f, 100.0f);
	}
	else
	{
		gluPerspective(45, (GLfloat)iWidth / (GLfloat)iHeight, 0.1f, 100.0f);
	}

	glViewport(0, 0, iWidth, iHeight);
}

void uninitialize()
{

}
