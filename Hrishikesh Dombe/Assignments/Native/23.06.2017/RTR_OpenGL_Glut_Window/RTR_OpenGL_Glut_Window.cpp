#include <GL/freeglut.h>
#include<gl/GL.h>

#pragma comment(lib,"User32.lib")
#pragma comment(lib,"GDI32.lib")
#pragma comment(lib,"opengl32.lib")

bool bFullscreen=false; 

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
	glutCreateWindow("OpenGL First Window : Hello World !!!");

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



	glutSwapBuffers();
}

void initialize(void)
{
	glClearColor(0.0f,0.0f,1.0f,0.0f);
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

}

void uninitialize(void)
{
	
}

