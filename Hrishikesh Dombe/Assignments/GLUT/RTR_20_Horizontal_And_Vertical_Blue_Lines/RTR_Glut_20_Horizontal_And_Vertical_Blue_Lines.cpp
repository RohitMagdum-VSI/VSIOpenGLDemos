#include <GL/freeglut.h>
#include<gl/GL.h>

#pragma comment(lib,"User32.lib")
#pragma comment(lib,"GDI32.lib")
#pragma comment(lib,"opengl32.lib")

bool bFullscreen=false; 
GLfloat gi_Y_Vertex, gi_X_Vertex;

int main(int argc,char** argv)
{
	void display(void);
	void resize(int,int);
	void keyboard(unsigned char,int,int);
	void mouse(int,int,int,int);
	void initialize(void);
	void uninitialize(void);

	glutInit(&argc,argv);

	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(800,600);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("OpenGL");

	initialize();

	glutDisplayFunc(display);
	glutReshapeFunc(resize);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutCloseFunc(uninitialize);

	glutMainLoop();

//	return(0); 
}

void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT);

	glLineWidth(3.0f);
	glBegin(GL_LINES);
	glColor3f(0.0f, 1.0f, 0.0f);
	glVertex3f(1.0f, 0.0f, 0.0f);
	glVertex3f(-1.0f, 0.0f, 0.0f);
	glColor3f(1.0f, 0.0f, 0.0f);
	glVertex3f(0.0f, 1.0f, 0.0f);
	glVertex3f(0.0f, -1.0f, 0.0f);
	glEnd();

	glLineWidth(1.0f);
	glBegin(GL_LINES);
	glColor3f(0.0f, 0.0f, 1.0f);
	for (gi_Y_Vertex = -1.0f; gi_Y_Vertex <= 0.0f; gi_Y_Vertex = gi_Y_Vertex + 0.05f)
	{
		glVertex3f(1.0f, gi_Y_Vertex, 0.0f);
		glVertex3f(-1.0f, gi_Y_Vertex, 0.0f);
	}
	for (gi_Y_Vertex = 0.05f; gi_Y_Vertex <= 1.0f; gi_Y_Vertex = gi_Y_Vertex + 0.05f)
	{
		glVertex3f(1.0f, gi_Y_Vertex, 0.0f);
		glVertex3f(-1.0f, gi_Y_Vertex, 0.0f);
	}
	for (gi_X_Vertex = -1.0f; gi_X_Vertex <= 0.0f; gi_X_Vertex = gi_X_Vertex + 0.05f)
	{
		glVertex3f(gi_X_Vertex, 1.0f, 0.0f);
		glVertex3f(gi_X_Vertex, -1.0f, 0.0f);
	}
	for (gi_X_Vertex = 0.05f; gi_X_Vertex <= 1.0f; gi_X_Vertex = gi_X_Vertex + 0.05f)
	{
		glVertex3f(gi_X_Vertex, 1.0f, 0.0f);
		glVertex3f(gi_X_Vertex, -1.0f, 0.0f);
	}
	glEnd();

	glutSwapBuffers();
}

void initialize(void)
{
	glClearColor(0.0f,0.0f,0.0f,0.0f);
}

void keyboard(unsigned char key,int x,int y)
{
	switch(key)
	{
	case 27:
		glutLeaveMainLoop();
		break;
	case 'F':
	case 'f':
		if(bFullscreen==false)
		{
			glutFullScreen();
			bFullscreen=true;
		}
		else
		{
			glutLeaveFullScreen();
			bFullscreen=false;
		}
		break;
	default:
		break;
	}
}

void mouse(int button,int state,int x,int y)
{
	switch(button)
	{
	case GLUT_LEFT_BUTTON:
		break;
	default:
		break;
	}
}

void resize(int width,int height)
{
	if (height == 0)
		height = 1;
	glViewport(0, 0, (GLsizei)width, (GLsizei)height);
}

void uninitialize(void)
{
	
}

