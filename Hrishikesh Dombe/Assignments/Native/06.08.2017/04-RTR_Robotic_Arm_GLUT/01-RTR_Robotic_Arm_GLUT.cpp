#include<C:\freeglut\include\GL\freeglut.h>
#include<stdlib.h>

#pragma comment(lib,"User32.lib")
#pragma comment(lib,"GDI32.lib")

static int shoulder = 0, elbow = 0;
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
	glPushMatrix();
	glTranslatef(-1.0f, 0.0f, 0.0f);
	glRotatef((GLfloat)shoulder, 0.0f, 0.0f, 1.0f);
	glTranslatef(1.0f, 0.0f, 0.0f);
	glPushMatrix();
	glScalef(2.0f, 0.4f, 1.0f);
	glutWireCube(1.0);
	glPopMatrix();

	glTranslatef(1.0f, 0.0f, 0.0f);
	glRotatef((GLfloat)elbow, 0.0f, 0.0f, 1.0f);
	glTranslatef(1.0f, 0.0f, 0.0f);
	glPushMatrix();
	glScalef(2.0f, 0.4f, 1.0f);
	glutWireCube(1.0);
	glPopMatrix();

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
	glTranslatef(0.0f, 0.0f, -15.0f);
}

void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 's':
		shoulder = (shoulder + 5) % 360;
		glutPostRedisplay();
		break;
	case 'S':
		shoulder = (shoulder - 5) % 360;
		glutPostRedisplay();
		break;
	case 'e':
		elbow = (elbow + 5) % 360;
		glutPostRedisplay();
		break;
	case 'E':
		elbow = (elbow - 5) % 360;
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