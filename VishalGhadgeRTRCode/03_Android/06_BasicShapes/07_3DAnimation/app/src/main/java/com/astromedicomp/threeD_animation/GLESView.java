package com.astromedicomp.threeD_animation;

import android.content.Context;
import android.opengl.GLSurfaceView;//	for open gl surface view all related.
import javax.microedition.khronos.opengles.GL10;
import javax.microedition.khronos.egl.EGLConfig;
import android.opengl.GLES30;	//	Change this version as per requirement.
import android.view.MotionEvent;
import android.view.GestureDetector;
import android.view.GestureDetector.OnGestureListener;
import android.view.GestureDetector.OnDoubleTapListener;

//	For VBO
import java.nio.ByteBuffer;	//	nio - non-blocking I/O
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import android.opengl.Matrix;

public class GLESView extends GLSurfaceView implements GLSurfaceView.Renderer,OnGestureListener,OnDoubleTapListener{

	private GestureDetector gestureDetector;
	private final Context context;
	
	private int shaderObjectVertex;
	private int shaderObjectFragment;
	private int shaderObjectProgram;
	
	private int[] VAOPyramid = new int[1];
	private int[] VAOCube = new int[1];
	private int[] VBOPosition = new int[1];
	private int[] VBOColor = new int[1];
	
	private float g_fAnglePyramid;
	private float g_fAngleCube;
	
	private float g_fAnimationSpeed;
	
	private int mvpUniform;
	
	private float perspectiveProjectionMatrix[] = new float[16];	//	4*4 Matrix.
	
    public GLESView(Context drawingContext)
	{
		super(drawingContext);
		context = drawingContext;
		
		g_fAnglePyramid = g_fAngleCube = 0.0f;
		g_fAnimationSpeed = 1.0f;
		
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
		//	Get GLSL version.
		String glslVersion = gl.glGetString(GLES30.GL_SHADING_LANGUAGE_VERSION);
		System.out.println("RTR: "+glslVersion);
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
		//	Increase animation speed.
		g_fAnimationSpeed = 2 * g_fAnimationSpeed;
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
		uninitialize();
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
		//---------------------------------------------------------------------------
		//+	Vertex shader 
		shaderObjectVertex = GLES30.glCreateShader(GLES30.GL_VERTEX_SHADER);
		
		//	Vertex shader source code.
		final String vertexShaderSourceCode = String.format
		(
		"#version 300 es"+
		"\n"+
		"in vec4 vPosition;"+
		"in vec4 vColor;"+
		"out vec4 out_color;"+
		"uniform mat4 u_mvp_matrix;"+
		"void main(void)"+
		"{"+
		"gl_Position = u_mvp_matrix * vPosition;"+
		"out_color = vColor;"+
		"}"
		);
		
		//	Provide source code to shader
		GLES30.glShaderSource(shaderObjectVertex,vertexShaderSourceCode);
		
		//	Compile shader and check for errors.
		GLES30.glCompileShader(shaderObjectVertex);
		
		int[] iShaderCompiledStatus = new int[1];
		int[] iInfoLogLength = new int[1];
		String szInfoLog = null;
		
		GLES30.glGetShaderiv(shaderObjectVertex,GLES30.GL_COMPILE_STATUS, iShaderCompiledStatus, 0);
		if (GLES30.GL_FALSE == iShaderCompiledStatus[0])
		{
			GLES30.glGetShaderiv(shaderObjectVertex, GLES30.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if (iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES30.glGetShaderInfoLog(shaderObjectVertex);
				System.out.println("RTR: Vertex shader compilation status Log: "+szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}
		//-	Vertex shader 
		//---------------------------------------------------------------------------
		
		//---------------------------------------------------------------------------
		//+	Fragment shader 
		shaderObjectFragment = GLES30.glCreateShader(GLES30.GL_FRAGMENT_SHADER);
		
		//	Vertex shader source code.
		final String fragmentShaderSourceCode = String.format
		(
		"#version 300 es"+
		"\n"+
		"precision highp float;"+
		"in vec4 out_color;"+
		"out vec4 FragColor;"+
		"void main(void)"+
		"{"+
		"FragColor = out_color;"+
		"}"
		);
		
		//	Provide source code to shader
		GLES30.glShaderSource(shaderObjectFragment,fragmentShaderSourceCode);
		
		//	Compile shader and check for errors.
		GLES30.glCompileShader(shaderObjectFragment);
		
		iShaderCompiledStatus[0] = 0;
		iInfoLogLength[0] = 0;
		szInfoLog = null;
		
		GLES30.glGetShaderiv(shaderObjectFragment,GLES30.GL_COMPILE_STATUS, iShaderCompiledStatus, 0);
		if (GLES30.GL_FALSE == iShaderCompiledStatus[0])
		{
			GLES30.glGetShaderiv(shaderObjectFragment, GLES30.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if (iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES30.glGetShaderInfoLog(shaderObjectFragment);
				System.out.println("RTR: Fragment shader compilation status Log: "+szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}
		//-	Fragment shader 
		//---------------------------------------------------------------------------

		//---------------------------------------------------------------------------
		//+	Shader program
		shaderObjectProgram = GLES30.glCreateProgram();
		
		GLES30.glAttachShader(shaderObjectProgram,shaderObjectVertex);
		
		GLES30.glAttachShader(shaderObjectProgram,shaderObjectFragment);
		
		//	pre-link binding of shader program object with vertex shader attributes.
		GLES30.glBindAttribLocation(shaderObjectProgram, GLESMacros.RTR_ATTRIBUTE_POSITION, "vPosition");
		GLES30.glBindAttribLocation(shaderObjectProgram, GLESMacros.RTR_ATTRIBUTE_COLOR, "vColor");
		
		//	Link the 2 shaders together to shader program
		GLES30.glLinkProgram(shaderObjectProgram);
		int[] iLinkStatus = new int[1];
		iInfoLogLength[0] = 0;
		szInfoLog = null;
		GLES30.glGetProgramiv(shaderObjectProgram, GLES30.GL_LINK_STATUS, iLinkStatus, 0);
		if (GLES30.GL_FALSE == iLinkStatus[0])
		{
			GLES30.glGetProgramiv(shaderObjectProgram, GLES30.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if (iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES30.glGetProgramInfoLog(shaderObjectProgram);
				System.out.println("RTR: Shader program Link log "+szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}
		//-	Shader program 
		//---------------------------------------------------------------------------
		
		//	Get MVP uniform location.
		mvpUniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_mvp_matrix");
		
		//	Vertices,colors,shader attributes,VBO,VAO initialization.
		final float arrPyramidVertices[] = new float[]
		{
			//	Front face
			0.0f, 1.0f, 0.0f,	//	apex
			-1.0f, -1.0f, 1.0f,	//	left_bottom
			1.0f, -1.0f, 1.0f,	//	right_bottom
			//	Right face
			0.0f, 1.0f, 0.0f,	//	apex
			1.0f, -1.0f, 1.0f,	//	left_bottom
			1.0f, -1.0f, -1.0f,	//	right_bottom
			//	Back face
			0.0f, 1.0f, 0.0f,	//	apex
			1.0f, -1.0f, -1.0f,	//	left_bottom
			-1.0f, -1.0f, -1.0f,	//	right_bottom
			//	Left face
			0.0f, 1.0f, 0.0f,	//	apex
			-1.0f, -1.0f, -1.0f,	//	left_bottom
			-1.0f, -1.0f, 1.0f,	//	right_bottom
		};
		
		//	Vertices,colors,shader attributes,VBO,VAO initialization.
		final float arrCubeVertices[] = new float[]
		{
			//	Front face
			1.0f, 1.0f, 1.0f,	//	left top
			1.0f, -1.0f, 1.0f,	//	left bottom
			-1.0f, -1.0f, 1.0f,	//	Right bottom
			-1.0f, 1.0f, 1.0f,	//	Right top

			//	Right face
			1.0f, 1.0f, -1.0f,	//	left top
			1.0f, 1.0f, 1.0f,	//	left bottom
			1.0f, -1.0f, 1.0f,	//	Right bottom
			1.0f, -1.0f, -1.0f,	//	Right top

			//	Top face
			1.0f, 1.0f, -1.0f,	//	left top
			-1.0f, 1.0f, -1.0f,	//	left bottom
			-1.0f, 1.0f, 1.0f,	//	Right bottom
			1.0f, 1.0f, 1.0f,	//	Right top

			//	Front face
			1.0f, 1.0f, -1.0f,	//	left top
			1.0f, -1.0f, -1.0f,	//	left bottom
			-1.0f, -1.0f, -1.0f,	//	Right bottom
			-1.0f, 1.0f, -1.0f,	//	Right top

			//	Right face
			-1.0f, 1.0f, -1.0f,	//	left top
			-1.0f, 1.0f, 1.0f,	//	left bottom
			-1.0f, -1.0f, 1.0f,	//	Right bottom
			-1.0f, -1.0f, -1.0f,	//	Right top

			//	Top face
			1.0f, -1.0f, -1.0f,	//	left top
			-1.0f, -1.0f, -1.0f,	//	left bottom
			-1.0f, -1.0f, 1.0f,	//	Right bottom
			1.0f, -1.0f, 1.0f,	//	Right top
		};
		
		final float arrPyramidColor[] =
		{
			//	Front face
			1.0f, 0.0f, 0.0f, 0.0f,//	Red
			0.0f, 1.0f, 0.0f, 0.0f,//	Green
			0.0f, 0.0f, 1.0f, 0.0f,//	Blue

			//	Front face
			1.0f, 0.0f, 0.0f, 0.0f,//	Red
			0.0f, 0.0f, 1.0f, 0.0f,//	Blue
			0.0f, 1.0f, 0.0f, 0.0f,//	Green

			//	Front face
			1.0f, 0.0f, 0.0f, 0.0f,//	Red
			0.0f, 1.0f, 0.0f, 0.0f,//	Green
			0.0f, 0.0f, 1.0f, 0.0f,//	Blue

			//	Front face
			1.0f, 0.0f, 0.0f, 0.0f,//	Red
			0.0f, 0.0f, 1.0f, 0.0f,//	Blue
			0.0f, 1.0f, 0.0f, 0.0f,//	Green
		};
		
		final float arrCubeColor[] =
		{
			1.0f, 0.0f, 0.0f, 0.0f,
			1.0f, 0.0f, 0.0f, 0.0f,
			1.0f, 0.0f, 0.0f, 0.0f,
			1.0f, 0.0f, 0.0f, 0.0f,
			
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			
			0.0f, 0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			
			1.0f, 1.0f, 0.0f, 0.0f,
			1.0f, 1.0f, 0.0f, 0.0f,
			1.0f, 1.0f, 0.0f, 0.0f,
			1.0f, 1.0f, 0.0f, 0.0f,
			
			0.0f, 1.0f, 1.0f, 0.0f,
			0.0f, 1.0f, 1.0f, 0.0f,
			0.0f, 1.0f, 1.0f, 0.0f,
			0.0f, 1.0f, 1.0f, 0.0f,
			
			1.0f, 0.0f, 1.0f, 0.0f,
			1.0f, 0.0f, 1.0f, 0.0f,
			1.0f, 0.0f, 1.0f, 0.0f,
			1.0f, 0.0f, 1.0f, 0.0f,
		};
		
		ByteBuffer byBuffer;
		FloatBuffer fVerticesBuffer;

		///////////////////////////////////////////////////////////////////
		//+	VAO Pyramid
		
		GLES30.glGenVertexArrays(1, VAOPyramid, 0);
		GLES30.glBindVertexArray(VAOPyramid[0]);
		
		///////////////////////////////////////////////////////////////////
		//+	VBO Position
		GLES30.glGenBuffers(1, VBOPosition, 0);
		GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, VBOPosition[0]);
		
		//	Convert array to byte data.
		byBuffer = ByteBuffer.allocateDirect(arrPyramidVertices.length * 4);
		byBuffer.order(ByteOrder.nativeOrder());
		fVerticesBuffer = byBuffer.asFloatBuffer();
		fVerticesBuffer.put(arrPyramidVertices);
		fVerticesBuffer.position(0);
		
		GLES30.glBufferData(
							GLES30.GL_ARRAY_BUFFER,
							arrPyramidVertices.length * 4,
							fVerticesBuffer,
							GLES30.GL_STATIC_DRAW
							);
							
		GLES30.glVertexAttribPointer(GLESMacros.RTR_ATTRIBUTE_POSITION, 3, GLES30.GL_FLOAT, false, 0, 0);
		
		GLES30.glEnableVertexAttribArray(GLESMacros.RTR_ATTRIBUTE_POSITION);
		
		GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0);
		//-	VBO Position
		///////////////////////////////////////////////////////////////////
		
		///////////////////////////////////////////////////////////////////
		//+	VBO Color
		GLES30.glGenBuffers(1, VBOColor, 0);
		GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, VBOColor[0]);
		
		//	Convert array to byte data.
		byBuffer = ByteBuffer.allocateDirect(arrPyramidColor.length * 4);
		byBuffer.order(ByteOrder.nativeOrder());
		fVerticesBuffer = byBuffer.asFloatBuffer();
		fVerticesBuffer.put(arrPyramidColor);
		fVerticesBuffer.position(0);
		
		GLES30.glBufferData(
							GLES30.GL_ARRAY_BUFFER,
							arrPyramidColor.length * 4,
							fVerticesBuffer,
							GLES30.GL_STATIC_DRAW
							);
							
		GLES30.glVertexAttribPointer(GLESMacros.RTR_ATTRIBUTE_COLOR, 4, GLES30.GL_FLOAT, false, 0, 0);
		
		GLES30.glEnableVertexAttribArray(GLESMacros.RTR_ATTRIBUTE_COLOR);
		
		GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0);
		//-	VBO Color
		///////////////////////////////////////////////////////////////////

		GLES30.glBindVertexArray(0);
		
		//-	VAO Triangle
		///////////////////////////////////////////////////////////////////
		
		///////////////////////////////////////////////////////////////////
		//+	VAO Rectangle
		
		GLES30.glGenVertexArrays(1, VAOCube, 0);
		GLES30.glBindVertexArray(VAOCube[0]);
		
		///////////////////////////////////////////////////////////////////
		//+	VBO Position
		GLES30.glGenBuffers(1, VBOPosition, 0);
		GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, VBOPosition[0]);
		
		//	Convert array to byte data.
		byBuffer = ByteBuffer.allocateDirect(arrCubeVertices.length * 4);
		byBuffer.order(ByteOrder.nativeOrder());
		fVerticesBuffer = byBuffer.asFloatBuffer();
		fVerticesBuffer.put(arrCubeVertices);
		fVerticesBuffer.position(0);
		
		GLES30.glBufferData(
							GLES30.GL_ARRAY_BUFFER,
							arrCubeVertices.length * 4,
							fVerticesBuffer,
							GLES30.GL_STATIC_DRAW
							);
							
		GLES30.glVertexAttribPointer(GLESMacros.RTR_ATTRIBUTE_POSITION, 3, GLES30.GL_FLOAT, false, 0, 0);
		
		GLES30.glEnableVertexAttribArray(GLESMacros.RTR_ATTRIBUTE_POSITION);
		
		GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0);
		//-	VBO Position
		///////////////////////////////////////////////////////////////////
		
		///////////////////////////////////////////////////////////////////
		//+	VBO Color
		GLES30.glGenBuffers(1, VBOColor, 0);
		GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, VBOColor[0]);
		
		//	Convert array to byte data.
		byBuffer = ByteBuffer.allocateDirect(arrCubeColor.length * 4);
		byBuffer.order(ByteOrder.nativeOrder());
		fVerticesBuffer = byBuffer.asFloatBuffer();
		fVerticesBuffer.put(arrCubeColor);
		fVerticesBuffer.position(0);
		
		GLES30.glBufferData(
							GLES30.GL_ARRAY_BUFFER,
							arrCubeColor.length * 4,
							fVerticesBuffer,
							GLES30.GL_STATIC_DRAW
							);
							
		GLES30.glVertexAttribPointer(GLESMacros.RTR_ATTRIBUTE_COLOR, 4, GLES30.GL_FLOAT, false, 0, 0);
		
		GLES30.glEnableVertexAttribArray(GLESMacros.RTR_ATTRIBUTE_COLOR);
		
		GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0);
		//-	VBO Color
		///////////////////////////////////////////////////////////////////
		GLES30.glBindVertexArray(0);
		
		//-	VAO Rectangle
		///////////////////////////////////////////////////////////////////
		
		//	Enable depth testing.
		GLES30.glEnable(GLES30.GL_DEPTH_TEST);
		//	dept test to do
		GLES30.glDepthFunc(GLES30.GL_LEQUAL);
		//	We will always cull backfaces for better performance
		//	Disable for animation
		//GLES30.glEnable(GLES30.GL_CULL_FACE);
		
		//	Set the background frame color.
		GLES30.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		
		//	Set Projection matrix to identity.
		Matrix.setIdentityM(perspectiveProjectionMatrix, 0);
	}
	
	private void resize(int width, int height)
	{
		if (height == 0)
			height = 1;
		
		if (width == 0)
			width = 1;

		//	Adjust the viewport based on geometry changes, 
		//	such as screen rotation.
		GLES30.glViewport(0, 0, width, height);
		if (width > height)
		{
			Matrix.perspectiveM(perspectiveProjectionMatrix, 0, 45.0f, (float)width/(float)height, 0.1f, 100.0f); 
		}
		else
		{
			Matrix.perspectiveM(perspectiveProjectionMatrix, 0, 45.0f, (float)height/(float)width, 0.1f, 100.0f); 
		}
	}
	
	public void draw()
	{
		//	Draw background color.
		GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT | GLES30.GL_DEPTH_BUFFER_BIT);
		
		//	Use shader program.
		GLES30.glUseProgram(shaderObjectProgram);
		
		//	OpenGL-ES drawing.
		float modelViewMatrix[] = new float[16];
		float modelViewProjectionMatrix[] = new float[16];

		////////////////////////////////////////////////////////////////////////////
		//+ Draw Pyramid
		Matrix.setIdentityM(modelViewMatrix, 0);
		Matrix.setIdentityM(modelViewProjectionMatrix, 0);
		
		Matrix.translateM(modelViewMatrix, 0, -2.0f,0.0f,-6.0f);
		
		Matrix.rotateM(modelViewMatrix, 0, g_fAnglePyramid,0.0f,1.0f,0.0f);
		
		//	Get modelViewProjectionMatrix
		//	MVPMatrix = ortho * mv matrix;
		Matrix.multiplyMM(modelViewProjectionMatrix, 0, perspectiveProjectionMatrix, 0, modelViewMatrix, 0);
		
		//	Pass mvp(above) matrix to vertex shader in 'u_mvp_matrix' shader variable.
		GLES30.glUniformMatrix4fv(mvpUniform, 1, false, modelViewProjectionMatrix, 0);
		
		//	Bind VAO
		GLES30.glBindVertexArray(VAOPyramid[0]);
		
		GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 12);
		
		// unbind VAO
		GLES30.glBindVertexArray(0);
		
		//- Draw Triangle
		////////////////////////////////////////////////////////////////////////////
		
		////////////////////////////////////////////////////////////////////////////
		//+ Draw Cube
		Matrix.setIdentityM(modelViewMatrix, 0);
		Matrix.setIdentityM(modelViewProjectionMatrix, 0);
		
		Matrix.translateM(modelViewMatrix, 0, 2.0f,0.0f,-6.0f);
		Matrix.scaleM(modelViewMatrix, 0, 0.8f,0.8f,0.8f);
		Matrix.rotateM(modelViewMatrix, 0, g_fAngleCube,1.0f,1.0f,1.0f);
		
		//	Get modelViewProjectionMatrix
		//	MVPMatrix = ortho * mv matrix;
		Matrix.multiplyMM(modelViewProjectionMatrix, 0, perspectiveProjectionMatrix, 0, modelViewMatrix, 0);
		
		//	Pass mvp(above) matrix to vertex shader in 'u_mvp_matrix' shader variable.
		GLES30.glUniformMatrix4fv(mvpUniform, 1, false, modelViewProjectionMatrix, 0);
		
		//	Bind VAO
		GLES30.glBindVertexArray(VAOCube[0]);
		
		GLES30.glDrawArrays(GLES30.GL_TRIANGLE_FAN, 0, 4);
		GLES30.glDrawArrays(GLES30.GL_TRIANGLE_FAN, 4, 4);
		GLES30.glDrawArrays(GLES30.GL_TRIANGLE_FAN, 8, 4);
		GLES30.glDrawArrays(GLES30.GL_TRIANGLE_FAN, 12, 4);
		GLES30.glDrawArrays(GLES30.GL_TRIANGLE_FAN, 16, 4);
		GLES30.glDrawArrays(GLES30.GL_TRIANGLE_FAN, 20, 4);
		
		
		// unbind VAO
		GLES30.glBindVertexArray(0);
		
		//- Draw Rectangle
		////////////////////////////////////////////////////////////////////////////
		
		
		//	un-use program.
		GLES30.glUseProgram(0);
		
		update();
		
		//	render/flush
		requestRender();	//	Same a SwapBuffers()
	}
	
	private void update()
	{
		g_fAnglePyramid = g_fAnglePyramid + g_fAnimationSpeed;
		if (g_fAnglePyramid > 360)
		{
			g_fAnglePyramid = 0.0f;
		}
		
		g_fAngleCube = g_fAngleCube + g_fAnimationSpeed;
		if (g_fAngleCube > 360)
		{
			g_fAngleCube = 0.0f;
		}
	}
	
	void uninitialize()
	{
		System.out.println("RTR:Uninitialize-->");
		if (VAOPyramid[0] != 0)
		{
			GLES30.glDeleteVertexArrays(1, VAOPyramid, 0);
			VAOPyramid[0] = 0;
		}
		
		if (VAOCube[0] != 0)
		{
			GLES30.glDeleteVertexArrays(1, VAOCube, 0);
			VAOCube[0] = 0;
		}
		
		if (VBOPosition[0] != 0)
		{
			GLES30.glDeleteBuffers(1, VBOPosition, 0);
			VBOPosition[0] = 0;
		}
		
		if (VBOColor[0] != 0)
		{
			GLES30.glDeleteBuffers(1, VBOColor, 0);
			VBOColor[0] = 0;
		}
		
		if (shaderObjectProgram != 0)
		{
			if (shaderObjectVertex != 0)
			{
				GLES30.glDetachShader(shaderObjectProgram, shaderObjectVertex);
				GLES30.glDeleteShader(shaderObjectVertex);
				shaderObjectVertex = 0;
			}
			if (shaderObjectFragment != 0)
			{
				GLES30.glDetachShader(shaderObjectProgram, shaderObjectFragment);
				GLES30.glDeleteShader(shaderObjectFragment);
				shaderObjectFragment = 0;
			}
		}
		
		if (shaderObjectProgram != 0)
		{
			GLES30.glDeleteProgram(shaderObjectProgram);
			shaderObjectProgram = 0;
		}
		System.out.println("RTR:Uninitialize<--");
	}
	//-	OpenGL methods
	//---------------------------------------------------------------------------
}
