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

#define WINDOW_CAPTION	"Solar System"

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

static int g_s_iYear;
static int g_s_iDay;

int main(void)
{
	//	Function prototype.
	void CreateWindow();
	void ToggleFullscreen();
	void uninitialize();
	void initialize();
	void display();
	void resize(int, int);
		
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

						case XK_d:
							g_s_iDay = (g_s_iDay + 6) % 360;
							break;
						case XK_D:
							g_s_iDay = (g_s_iDay - 6) % 360;
							break;

						case XK_y:
							g_s_iYear = (g_s_iYear + 3) % 360;
							break;
						case XK_Y:
							g_s_iYear = (g_s_iYear - 3) % 360;
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
	GLUquadric *gluQuadric = NULL;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	//	Change 3 for 3D

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	//
	//	View transaformation.
	//
	gluLookAt(0.0f, 0.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);

	glPushMatrix();
		glRotatef(90.0f, 1.0f, 0.0f, 0.0f);
		//
		//	Draw sun.
		//
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

		gluQuadric = gluNewQuadric();
		glColor3f(1.0f, 1.0f, 0.0f);
		gluSphere(gluQuadric, 0.75, 40, 40);
	glPopMatrix();

	glPushMatrix();
		glRotatef((GLfloat)g_s_iYear, 0.0f, 1.0f, 0.0f);
		glTranslatef(1.5f, 0.0f, 0.0f);

		glRotatef(90.0f, 1.0f, 0.0f, 0.0f);

		glRotatef((GLfloat)g_s_iDay, 0.0f, 0.0f, 1.0f);

		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

		gluQuadric = gluNewQuadric();
		glColor3f(0.4f, 0.9f, 1.0f);
		gluSphere(gluQuadric, 0.2, 30, 30);
	glPopMatrix();
	
	glXSwapBuffers(g_pDisplay, g_Window);
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
