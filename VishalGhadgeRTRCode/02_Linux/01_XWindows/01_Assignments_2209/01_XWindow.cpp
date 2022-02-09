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

//	Namespaces.
using namespace std;

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

int main(void)
{
	//	Function prototype.
	void CreateWindow();
	void ToggleFullscreen();
	void uninitialize();
	
	//	Variable declaration.
	int iWinWidth = g_iWindowWidth;
	int iWinHeight = g_iWindowHeight;
	
	CreateWindow();
	
	//
	//	Message loop.
	//
	XEvent xEvent;
	KeySym keysym;
	
	while (1)
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
							uninitialize();
							exit(1);
							
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
	
	return 0;
}


void CreateWindow()
{
	void uninitialize();
	
	XSetWindowAttributes WindowAttributes;
	int iDefaultScreen;
	int iDefaultDepth;
	int iStyleMask;
	
	printf("==>Create Window");
	
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
	g_pXVisualInfo = (XVisualInfo *) malloc (sizeof(XVisualInfo));
	if (NULL == g_pXVisualInfo)
	{
		printf("\n malloc(g_pXVisualInfo) failed");
		uninitialize();
		exit(1);
	}
	
	//
	//	Allocated g_pXVisualInfo structure will fill up by below API. 
	//
	if (0 == XMatchVisualInfo(g_pDisplay, iDefaultScreen, iDefaultDepth, TrueColor, g_pXVisualInfo))	
	{
		printf("\n XMatchVisualInfo failed");
		uninitialize();
		exit(1);
	}
	
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
	XStoreName(g_pDisplay, g_Window, "First XWindow");
	
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
	
	printf("<==Create Window");
}


void ToggleFullscreen()
{
	Atom wm_state;
	Atom fullscreen;
	XEvent xev = {0};
	
	printf("\n==>ToggleFullscreen");
	
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
		
	printf("\n<==ToggleFullscreen");
}


void uninitialize()
{
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