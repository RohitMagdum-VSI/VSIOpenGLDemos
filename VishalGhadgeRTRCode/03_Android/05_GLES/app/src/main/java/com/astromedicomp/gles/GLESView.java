package com.astromedicomp.gles;

import android.content.Context;
import android.opengl.GLSurfaceView;//	for open gl surface view all related.
import javax.microedition.khronos.opengles.GL10;
import javax.microedition.khronos.egl.EGLConfig;
import android.opengl.GLES30;	//	Change this version as per requirement.
import android.view.MotionEvent;
import android.view.GestureDetector;
import android.view.GestureDetector.OnGestureListener;
import android.view.GestureDetector.OnDoubleTapListener;

public class GLESView extends GLSurfaceView implements GLSurfaceView.Renderer,OnGestureListener,OnDoubleTapListener{

	private GestureDetector gestureDetector;
	private final Context context;
	
    public GLESView(Context drawingContext)
	{
		super(drawingContext);
		context = drawingContext;
		
		//
		//	Accordingly set EGLContext to supported version of OpenGL-ES
		//
		setEGLContextClientVersion(3);
		
		//
		//	Set renderer for drawing on the GLSurfaceView.
		//
		setRenderer(this);	//	Because of this OnSurfaceCreated() get called.
		
		//
		//	Render the view only when there is change in the drawing data.
		//
		setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
		
		
		gestureDetector = new GestureDetector(context,this, null, false);
		gestureDetector.setOnDoubleTapListener(this);	//	this means handler i.e) who is going to handle.
	}
	
	//---------------------------------------------------------------------------
	//+	Overriden methods of GLSurfaceView.Renderer
	
	//
	//	Init code
	//
	@Override
	public void onSurfaceCreated(GL10 gl, EGLConfig config)
	{
		//	OpenGL-ES version check.
		String version = gl.glGetString(GL10.GL_VERSION);
		System.out.println("RTR: "+version);
		initialize(gl);
	}
	
	//
	//	Change size code.
	//
	@Override
	public void onSurfaceChanged(GL10 unused, int width, int height)
	{
		resize(width, height);
	}
	
	//
	//	Rendering code.
	//
	@Override
	public void onDrawFrame(GL10 unused)
	{
		draw();
	}
	//-	Overriden methods of GLSurfaceView.Renderer
	//---------------------------------------------------------------------------
	
	//
	//	Handling 'onTouchEvent' is the most IMPORTANT.
	//	Because it triggers all gesture and tap events.
	//
	@Override
	public boolean onTouchEvent(MotionEvent event)
	{
		int eventAction = event.getAction();
		if (!gestureDetector.onTouchEvent(event))
		{
			super.onTouchEvent(event);
		}
		return(true);
	}
	
	//---------------------------------------------------------------------------
	//+	Abstract method from OnDoubleTapListener
	
	//
	//	onDoubleTap
	//
	@Override
	public boolean onDoubleTap(MotionEvent e)
	{
		System.out.println("RTR:Double Tap");
		return(true);
	}
	
	//
	//	onDoubleTapEvent
	//
	@Override
	public boolean onDoubleTapEvent(MotionEvent e)
	{
		//	Do not write any code here bacause already written 'onDoubleTap'
		return true;
	}
	
	//
	//	onSingleTapConfirmed
	//
	@Override
	public boolean onSingleTapConfirmed(MotionEvent e)
	{
		System.out.println("RTR:Single Tap");
		return true;
	}
	
	//
	//	onDown
	//
	@Override
	public boolean onDown(MotionEvent e)
	{
		//	Do not write any code here bacause already written 'onSingleTapConfirmed'
		return true;
	}
	//-	Abstract method from OnDoubleTapListener
	//---------------------------------------------------------------------------
	
	//---------------------------------------------------------------------------
	//+	abstract method from OnGestureListener so must be implemented
	
	//
	//	onFling
	//
	@Override
	public boolean onFling(MotionEvent e1, MotionEvent e2, float velocityX, float velocityY)
	{
		System.out.println("RTR:Flingg");
		return true;
	}
	
	//
	//	onLongPress
	//
	@Override
	public void onLongPress(MotionEvent e)
	{
		System.out.println("RTR:Long press");
	}
	
	//
	//	onScroll
	//
	@Override
	public boolean onScroll(MotionEvent e1,MotionEvent e2, float distanceX, float distanceY)
	{
		System.out.println("RTR:Scroll");
		System.exit(0);
		return true;
	}
	
	//
	//	onShowPress
	//
	@Override
	public void onShowPress(MotionEvent e)
	{
		
	}
	
	//
	//	onSingleTapUp
	//
	@Override
	public boolean onSingleTapUp(MotionEvent e)
	{
		return true;
	}
	//-	abstract method from OnGestureListener so must be implemented
	//---------------------------------------------------------------------------
	
	//---------------------------------------------------------------------------
	//+	OpenGL methods
	
	private void initialize(GL10 gl)
	{
		//	Set the background frame color.
		GLES30.glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
	}
	
	private void resize(int width, int height)
	{
		//	Adjust the viewport based on geometry changes, 
		//	such as screen rotation.
		GLES30.glViewport(0, 0, width, height);
	}
	
	public void draw()
	{
		//	Draw background color.
		GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT | GLES30.GL_DEPTH_BUFFER_BIT);
	}
	//-	OpenGL methods
	//---------------------------------------------------------------------------
}
