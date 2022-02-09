#include<C:\\freeglut\include\GL\freeglut.h>
#include<stdlib.h>

static int year = 0, day = 0;
bool gbFullscreen = false;

int main(int argc, char** argv)
{
	void init(void);
	void display(void);
	void resize(int, int);
	void keyboard(unsigned char, int, int);
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(500, 500);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("OpenGL Using GLUT");
	init();
	glutDisplayFunc(display);
	glutReshapeFunc(resize);
	glutKeyboardFunc(keyboard);
	glutMainLoop();
}

void init(void)
{
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_FLAT);
}

void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1.0f, 1.0f, 1.0f);

	glPushMatrix();
	glutWireSphere(1.0, 20, 16);
	glRotatef((GLfloat)year, 0.0f, 1.0f, 0.0f);
	glTranslatef(2.0f, 0.0f, 0.0f);
	glRotatef((GLfloat)day, 0.0f, 1.0f, 0.0f);
	glutWireSphere(0.2, 10, 8);
	glPopMatrix();
	glutSwapBuffers();
}

void resize(int width, int height)
{
	glViewport(0, 0, (GLsizei)width, (GLsizei)height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
}

void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 'd':
		day = (day + 10) % 360;
		glutPostRedisplay();
		break;
	case 'D':
		day = (day - 10) % 360;
		glutPostRedisplay();
		break;
	case 'y':
		year = (year + 5) % 360;
		glutPostRedisplay();
		break;
	case 'Y':
		year = (year - 5) % 360;
		glutPostRedisplay();
		break;
	case 'f':
	case 'F':
		if (gbFullscreen == false)
		{
			glutFullScreen();
			gbFullscreen = true;
		}
		else
		{
			glutLeaveFullScreen();
			gbFullscreen = false;
		}
		break;
	case 27:
		glutLeaveMainLoop();
		break;
	default:
		break;
	}
}