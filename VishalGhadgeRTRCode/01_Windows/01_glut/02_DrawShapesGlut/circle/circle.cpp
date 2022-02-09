#include <GL\freeglut.h>
#include <math.h>

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

	glutCreateWindow("GLUT- Circle");

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
	VOID DrawGrid();
	VOID DrawCircle();

	glClear(GL_COLOR_BUFFER_BIT);

	DrawGrid();
	DrawCircle();

	glutSwapBuffers();
}


VOID DrawCircle()
{
	const float PI = 3.14f;

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glLineWidth(1.0f);
	//
	//	Circle.
	//
	glColor3f(1.0f, 1.0f, 0.0f);
	glBegin(GL_POINTS);
	for (GLfloat glfAngle = 0.0f; glfAngle < 2.0f * PI; glfAngle += 0.001f)
	{
		glVertex3f(cos(glfAngle), sin(glfAngle), 0.0f);
	}
	glEnd();
}


VOID DrawGrid()
{
	GLfloat glfVertex;

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glfVertex = 1.0f;

	glLineWidth(1.0f);
	glColor3f(0.0f, 0.0f, 1.0f);
	for (GLfloat glfGraphVertex = 0.05f; glfGraphVertex <= 1.0f; glfGraphVertex += 0.05f)
	{
		//
		//	Vertical lines in Right side.
		//
		glBegin(GL_LINES);
		glVertex3f(glfGraphVertex, glfVertex, 0.0f);
		glVertex3f(glfGraphVertex, -glfVertex, 0.0f);
		glEnd();

		//
		//	Vertical lines in left side.
		//
		glBegin(GL_LINES);
		glVertex3f(-glfGraphVertex, glfVertex, 0.0f);
		glVertex3f(-glfGraphVertex, -glfVertex, 0.0f);
		glEnd();

		////
		////	Horizontal Lines in top side.
		////
		glBegin(GL_LINES);
		glVertex3f(-glfVertex, glfGraphVertex, 0.0f);
		glVertex3f(glfVertex, glfGraphVertex, 0.0f);
		glEnd();

		////
		////	Horizontal Lines in bottom side.
		////
		glBegin(GL_LINES);
		glVertex3f(-glfVertex, -glfGraphVertex, 0.0f);
		glVertex3f(glfVertex, -glfGraphVertex, 0.0f);
		glEnd();
	}

	glLineWidth(3.0f);

	glColor3f(1.0f, 0.0f, 0.0f);
	//	Horizontal Line
	glBegin(GL_LINES);
	glVertex3f(glfVertex, 0.0f, 0.0f);
	glVertex3f(-glfVertex, 0.0f, 0.0f);
	glEnd();

	glColor3f(0.0f, 1.0f, 0.0f);
	//	Vertical Line
	glBegin(GL_LINES);
	glVertex3f(0.0f, glfVertex, 0.0f);
	glVertex3f(0.0f, -glfVertex, 0.0f);
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
