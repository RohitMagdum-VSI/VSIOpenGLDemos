#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

//
//	XServer Files.
//
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/XKBlib.h>
#include <X11/keysym.h>

#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glu.h>

//	Namespaces.
using namespace std;

#define WINDOW_CAPTION	"Rotating Shapes 3D"

//
//	Gloabal Declaration.
//
bool g_bFullScreen = false;
Display *g_pDisplay = NULL;
XVisualInfo *g_pXVisualInfo = NULL;
Colormap g_ColorMap;
Window g_Window;
int g_iWindowWidth = 800;
int g_iWindowHeight = 600;

GLXContext g_GLXContext;	//	same as HGLRC in windows.

float g_fPyramidAngle = -360.0f;
float g_fCubeAngle = -360.0f;
#define SPEED 1.0f

int main(void)
{
	//	Function prototype.
	void CreateWindow();
	void ToggleFullscreen();
	void uninitialize();
	void initialize();
	void display();
	void resize(int, int);
	void update();
	
	//	Variable declaration.
	int iWinWidth = g_iWindowWidth;
	int iWinHeight = g_iWindowHeight;
	bool bDone = false;
	
	CreateWindow();
	
	//	Initialize.
	initialize();
	
	//
	//	Message loop.
	//
	XEvent xEvent;
	KeySym keysym;
	
	while (false == bDone)
	{
		while (XPending(g_pDisplay))
		{
			XNextEvent(g_pDisplay, &xEvent);
			switch (xEvent.type)
			{
				case MapNotify: //	Similar to WM_CREATE
					break;
				
				case KeyPress:
				
					keysym = XkbKeycodeToKeysym(
												g_pDisplay,
												xEvent.xkey.keycode,
												0,	//	Key symbol group level.
												0	//	Local (default)
												);
					switch(keysym)
					{
						case XK_Escape:
								bDone = true;
								break;
							
						case XK_F:
						case XK_f:
								//	Fullscreen.
								if (false == g_bFullScreen)
								{
									ToggleFullscreen();
									g_bFullScreen = true;
								}
								else
								{
									ToggleFullscreen();
									g_bFullScreen = false;
								}
							
								break;
							
						default:
								break;
					}
				
					break;
				
				case ButtonPress:
				
					switch(xEvent.xbutton.button)
					{
						case 1:	//	WM_LButtonDown - Left
							
								break;
							
						case 2:	//	WM_MButtonDown - Middle
							
								break;
							
						case 3:	//	WM_RButtonDown - Right
							
								break;
							
						default:
								break;
					}
				
					break;
				
				case MotionNotify:	//	Move mouse - WM_MOUSEMOVE
				
					break;
				
				case ConfigureNotify:	//	WM_SIZE
				
					g_iWindowWidth = xEvent.xconfigure.width;	//	LOWORD(LPARAM)
					g_iWindowHeight = xEvent.xconfigure.height;	//	HIWORD(LPARAM)
					
					resize(g_iWindowWidth, g_iWindowHeight);
					break;
				
				case Expose:	//	WM_PAINT
				
					break;
				
				case DestroyNotify:	//	WM_DESTROY
				
					break;
				
				case 33:	//	Close button or exit system menu - close (WM_DELETE_WINDOW in XInternAtom)
			
					uninitialize();
					exit(0);
					break;
				
				default:
				
					break;
			}
		}
		
		update();
		display();
	}
	
	return 0;
}


void CreateWindow()
{
	void uninitialize();
	
	XSetWindowAttributes WindowAttributes;
	int iDefaultScreen;
	int iDefaultDepth;
	int iStyleMask;
	
	static int iFrameBufferAttributes[] =
	{
		GLX_RGBA,
		GLX_RED_SIZE, 8,	//	8 is for double buffer.
		GLX_GREEN_SIZE, 8,
		GLX_BLUE_SIZE, 8,
		GLX_ALPHA_SIZE, 8,
		GLX_DOUBLEBUFFER, True,	//	Added for double buffer.
		GLX_DEPTH_SIZE, 24,
		None	//	Stopper of array
	};
	
	g_pDisplay = XOpenDisplay(NULL);//	It will connect with XServer and return display structure pointer.	
	if (NULL == g_pDisplay)
	{
		printf("\n XOpenDisplay failed");
		uninitialize();
		exit(1);
	}

	//
	//	It return index of primary monitor. 
	//	
	iDefaultScreen = XDefaultScreen(g_pDisplay);

	//
	//	Bits per color = Depth.
	//	
	iDefaultDepth = XDefaultDepth(g_pDisplay, iDefaultScreen);

	//
	//	Get Visual of imediate mode and allocate size.
	//
	g_pXVisualInfo = glXChooseVisual(g_pDisplay, iDefaultScreen, iFrameBufferAttributes);
	
	WindowAttributes.border_pixel = 0;
	WindowAttributes.background_pixmap = 0;
	WindowAttributes.colormap = XCreateColormap(
											g_pDisplay,
											RootWindow(g_pDisplay, g_pXVisualInfo->screen),	//	Similar to HWND_DESKTOP(Parent Window)
											g_pXVisualInfo->visual,
											AllocNone	//	Child window will share memory, allocate when child window require 
											);			//	seperate memory.
											
	g_ColorMap = WindowAttributes.colormap;	//	Will use in Open GL.
	
	WindowAttributes.background_pixel = BlackPixel(g_pDisplay, iDefaultScreen);	//	Similar to hbrBackground
	
	WindowAttributes.event_mask = 	ExposureMask |			//	Expose(Paint)
									VisibilityChangeMask |	//	WS_VISIBLE
									ButtonPressMask	|		//	ButtonPress
									KeyPressMask |			//	KeyPress
									PointerMotionMask |		//	PinterMotionNotify
									StructureNotifyMask;	//	MapNotify, ConfigureNotify, DestroyNotify.
									
	iStyleMask = CWBorderPixel | CWBackPixel | CWEventMask | CWColormap;
	
	g_Window = XCreateWindow(
							g_pDisplay,
							RootWindow(g_pDisplay, g_pXVisualInfo->screen),
							0, //	X-Co-ordinate
							0, //	y-Co-ordinate
							g_iWindowWidth,
							g_iWindowHeight,
							0,
							g_pXVisualInfo->depth, //	bit depth.
							InputOutput, //	Type of window. (InputOutput, InputOnly, utputOnly)
							g_pXVisualInfo->visual,
							iStyleMask,
							&WindowAttributes
							);
	if (!g_Window)
	{
		printf("\n XCreateWindow failed");
		uninitialize();
		exit(1);
	}
	
	//
	//	Add caption to window.
	//
	XStoreName(g_pDisplay, g_Window, WINDOW_CAPTION);
	
	Atom WindowManagerDelete = XInternAtom(
										g_pDisplay,
										"WM_DELETE_WINDOW",	// Protocol from xlib specification.
										True	//	TRUE - Create Always, False - If not already created then create.
										);

	XSetWMProtocols(
					g_pDisplay,
					g_Window,
					&WindowManagerDelete,	// Can be array of Atom's.
					1	//	No of array elements.
					);
				
	XMapWindow(g_pDisplay, g_Window);	//	Similar to ShowWindow() And UpdateWindow()
}


void ToggleFullscreen()
{
	Atom wm_state;
	Atom fullscreen;
	XEvent xev = {0};
	
	wm_state = XInternAtom(g_pDisplay, "_NET_WM_STATE", False);
	
	memset(&xev, 0, sizeof(xev));
	
	xev.type = ClientMessage;	//	WM_USER(User defined event)
	xev.xclient.window = g_Window;
	xev.xclient.message_type = wm_state;
	xev.xclient.format = 32;
	//
	//	"data" is union of below members:
	//	char b20
	//	short s10
	//	long l5
	xev.xclient.data.l[0] = g_bFullScreen ? 0 : 1;
	
	fullscreen = XInternAtom(g_pDisplay, "_NET_WM_STATE_FULLSCREEN", False);
	
	xev.xclient.data.l[1] = fullscreen;
	
	XSendEvent(
			g_pDisplay,
			RootWindow(g_pDisplay, g_pXVisualInfo->screen),
			False,	//	Do not send this message to siblings.
			StructureNotifyMask,	//	Resizing mask.
			&xev
			);
}


void initialize()
{
	void resize(int, int);
	
	g_GLXContext = glXCreateContext(
									g_pDisplay,
									g_pXVisualInfo,
									NULL,	//	shareble context, Multi Monitor application(eg. Fish tank) 
									GL_TRUE	//	Hardware rendering - true, Software rendering - false.
									 );
									 
	glXMakeCurrent(g_pDisplay, g_Window, g_GLXContext);
	
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	
	//+	Change 2 For 3D
	glClearDepth(1.0f);

	glEnable(GL_DEPTH_TEST);

	glDepthFunc(GL_LEQUAL);

	//
	//	Optional.
	//
	glShadeModel(GL_SMOOTH);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	//-	Change 2 For 3D
	
	resize(g_iWindowWidth, g_iWindowHeight);
}


void display()
{
	void MultiColorPyramid();
	void DrawMultiColorCube();
	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	
	glTranslatef(-3.0f, 0.0f, -8.0f);
	glRotatef(g_fPyramidAngle, 0.0f, 1.0f, 0.0f);
	MultiColorPyramid();
	
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	
	glTranslatef(3.0f, 0.0f, -8.0f);
	glRotatef(g_fCubeAngle, 0.0f, 1.0f, 0.0f);	
	DrawMultiColorCube();
	
	glXSwapBuffers(g_pDisplay, g_Window);
}


void MultiColorPyramid()
{
	glBegin(GL_TRIANGLES);
	
	//	Front face
	glColor3f(1.0f, 0.0f, 0.0f);
	glVertex3f(0.0f, 1.0f, 0.0f);
	glVertex3f(-1.0f, -1.0f, 1.0f);
	glVertex3f(1.0f, -1.0f, 1.0f);
	
	//	Back face
	glColor3f(1.0f, 0.0f, 0.0f);
	glVertex3f(0.0f, 1.0f, 0.0f);
	glVertex3f(-1.0f, -1.0f, -1.0f);
	glVertex3f(1.0f, -1.0f, -1.0f);

	//	Left face
	glColor3f(1.0f, 0.0f, 0.0f);
	glVertex3f(0.0f, 1.0f, 0.0f);
	glVertex3f(-1.0f, -1.0f, -1.0f);
	glVertex3f(-1.0f, -1.0f, 1.0f);
	
	//	Right face
	glColor3f(1.0f, 0.0f, 0.0f);
	glVertex3f(0.0f, 1.0f, 0.0f);
	glVertex3f(1.0f, -1.0f, -1.0f);
	glVertex3f(1.0f, -1.0f, 1.0f);
	
	glEnd();
	
	glBegin(GL_QUADS);
	glColor3f(1.0f, 0.0f, 0.0f);
	glVertex3f(-1.0f, -1.0f, 1.0f);
	glVertex3f(1.0f, -1.0f, 1.0f);
	glVertex3f(1.0f, -1.0f, -1.0f);	
	glVertex3f(-1.0f, -1.0f, -1.0f);
	glEnd();
}


void DrawMultiColorCube()
{
	glBegin(GL_QUADS);
	//Front face.
	glColor3f(1.0f, 0.0f, 0.0f);	//	Red
	glVertex3f(1.0f, 1.0f, 1.0f);
	glVertex3f(1.0f, -1.0f, 1.0f);
	glVertex3f(-1.0f, -1.0f, 1.0f);
	glVertex3f(-1.0f, 1.0f, 1.0f);

	//	Right face
	glColor3f(0.0f, 0.0f, 1.0f);	//	Blue
	glVertex3f(1.0f, 1.0f, -1.0f);
	glVertex3f(1.0f, 1.0f, 1.0f);
	glVertex3f(1.0f, -1.0f, 1.0f);
	glVertex3f(1.0f, -1.0f, -1.0f);


	//	Top face
	glColor3f(0.0f, 1.0f, 0.0f);	//	Green
	glVertex3f(1.0f, 1.0f, -1.0f);
	glVertex3f(-1.0f, 1.0f, -1.0f);
	glVertex3f(-1.0f, 1.0f, 1.0f);
	glVertex3f(1.0f, 1.0f, 1.0f);

	//	Back face.
	glColor3f(1.0f, 1.0f, 0.0f);
	glVertex3f(1.0f, 1.0f, -1.0f);
	glVertex3f(1.0f, -1.0f, -1.0f);
	glVertex3f(-1.0f, -1.0f, -1.0f);
	glVertex3f(-1.0f, 1.0f, -1.0f);

	//	Left face
	glColor3f(0.0f, 1.0f, 1.0f);
	glVertex3f(-1.0f, 1.0f, -1.0f);
	glVertex3f(-1.0f, 1.0f, 1.0f);
	glVertex3f(-1.0f, -1.0f, 1.0f);
	glVertex3f(-1.0f, -1.0f, -1.0f);

	//	Bottom face
	glColor3f(1.0f, 0.0f, 1.0f);
	glVertex3f(1.0f, -1.0f, -1.0f);
	glVertex3f(-1.0f, -1.0f, -1.0f);
	glVertex3f(-1.0f, -1.0f, 1.0f);
	glVertex3f(1.0f, -1.0f, 1.0f);

	glEnd();
}


void update()
{
	if (g_fPyramidAngle <= 360.0f)
	{
		g_fPyramidAngle += SPEED;	
	}
	else
	{
		g_fPyramidAngle = -360.0f;
	}
	
	if (g_fCubeAngle <= 360.0f)
	{
		g_fCubeAngle += SPEED;	
	}
	else
	{
		g_fCubeAngle = -360.0f;
	}
}

void resize(int iWidth, int iHeight)
{
	if (0 == iHeight)
		iHeight = 1;
		
	if (0 == iWidth)
		iWidth = 1;
		
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	
	if (iWidth > iHeight)
	{
		gluPerspective(45.0f, (GLfloat)iWidth / (GLfloat)iHeight, 0.1f, 100.0f);
	}
	else
	{
		gluPerspective(45.0f, (GLfloat)iHeight / (GLfloat)iWidth, 0.1f, 100.0f);
	}
	
	glViewport(0, 0, (GLsizei)iWidth, (GLsizei)iHeight);
}


void uninitialize()
{
	GLXContext GLXCurrentContext;
	
	GLXCurrentContext = glXGetCurrentContext();
	
	if (NULL != GLXCurrentContext && GLXCurrentContext != g_GLXContext)
	{
		glXMakeCurrent(g_pDisplay, 0, 0);	//	To handle multi monitor scenario.
	}
	
	if (g_GLXContext)
	{
		glXDestroyContext(g_pDisplay, g_GLXContext);
	}
	
	if (g_Window)
	{
		XDestroyWindow(g_pDisplay, g_Window);
	}
	
	if (g_ColorMap)
	{
		free(g_pXVisualInfo);
		g_pXVisualInfo = NULL;
	}
	
	if (g_pDisplay)
	{
		XCloseDisplay(g_pDisplay);
		g_pDisplay = NULL;
	}
}
