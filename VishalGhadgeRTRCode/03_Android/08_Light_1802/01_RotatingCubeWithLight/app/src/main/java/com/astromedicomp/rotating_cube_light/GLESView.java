package com.astromedicomp.rotating_cube_light;

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
	
	private int[] VAO = new int[1];
	private int[] VBOPosition = new int[1];
	private int[] VBONormal = new int[1];
	private int g_iModelViewMat4Uniform;
	private int g_iProjectionMat4Uniform;
	private int g_iLightPositionVec4Uniform;
	private int g_iLDVec3Uniform;
	private int g_iKDVec3Uniform;
	private int g_iDoubleTapUniform;
		
	private float perspectiveProjectionMatrix[] = new float[16];	//	4*4 Matrix.
	
	private float g_fAngleCube = 0.0f;
	
	private int g_iSingleTap;	// For Animation
	private int g_iDoubleTap;	// For Light 
	
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
		
		g_iDoubleTap++;
		if (g_iDoubleTap > 1)
		{
			g_iDoubleTap = 0;
		}
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
		
		g_iSingleTap++;
		if (g_iSingleTap > 1)
		{
			g_iSingleTap = 0;
		}
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
		"in vec3 vNormal;"+
		"uniform mat4 u_model_view_matrix;"+
		"uniform mat4 u_projection_matrix;"+
		"uniform mediump int u_double_tap;"+
		"uniform vec3 u_LD;"+
		"uniform vec3 u_KD;"+
		"uniform vec4 u_light_position;"+
		"out vec3 out_diffuse_light;"+
		"void main(void)"+
		"{"+
			"if (1 == u_double_tap)"+
			"{"+
				"vec4 eyeCoordinates = u_model_view_matrix * vPosition;"+
				"vec3 tnorm = normalize(mat3(u_model_view_matrix) * vNormal);"+
				"vec3 srcVec = normalize(vec3(u_light_position - eyeCoordinates));"+
				"out_diffuse_light = u_LD * u_KD * max(dot(srcVec, tnorm), 0.0);"+
			"}"+
		"gl_Position = u_projection_matrix * u_model_view_matrix * vPosition;"+
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
		"in vec3 out_diffuse_light;"+
		"uniform int u_double_tap;"+
		"out vec4 vFragColor;"+
		"void main(void)"+
		"{"+
			"vec4 color;"+
			"if (1 == u_double_tap)"+
			"{"+
				"color = vec4(out_diffuse_light, 1.0);"+
			"}"+
			"else"+
			"{"+
				"color = vec4(1.0, 1.0, 1.0, 1.0);"+
			"}"+
			"vFragColor = color;"+
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
		GLES30.glBindAttribLocation(shaderObjectProgram, GLESMacros.RTR_ATTRIBUTE_NORMAL, "vNormal");
		
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
		
		//	Get uniform location's.
		g_iModelViewMat4Uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_model_view_matrix");
		g_iProjectionMat4Uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_projection_matrix");
		g_iDoubleTapUniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_double_tap");
		g_iLDVec3Uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_LD");
		g_iKDVec3Uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_KD");
		g_iLightPositionVec4Uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_light_position");
		
		//	Vertices,colors,shader attributes,VBO,VAO initialization.
		final float farrCubeVertices[] = new float[]
		{
			//	Top face
			1.0f, 1.0f, -1.0f,	//	top right 
			-1.0f, 1.0f, -1.0f,	//	top left 
			-1.0f, 1.0f, 1.0f,	//	bottom left
			1.0f, 1.0f, 1.0f,	//	bottom right 

			//	Bottom face
			1.0f, -1.0f, 1.0f,//	top right 
			-1.0f, -1.0f, 1.0f,	//	top left 
			-1.0f, -1.0f, -1.0f,	//	bottom left 
			1.0f, -1.0f, -1.0f,	//	bottom right

			//	Front face
			1.0f, 1.0f, 1.0f,	//	top right 
			-1.0f, 1.0f, 1.0f,	//	top left 
			-1.0f, -1.0f, 1.0f,	//	bottom left 
			1.0f, -1.0f, 1.0f,	//	bottom Right 

			//	Back face
			1.0f, -1.0f, -1.0f,	//	top Right 
			-1.0f, -1.0f, -1.0f,//	top left 
			-1.0f, 1.0f, -1.0f,	//	bottom left 
			1.0f, 1.0f, -1.0f,	//	bottom right 

			//	Left face
			-1.0f, 1.0f, 1.0f,	//	top right 
			-1.0f, 1.0f, -1.0f,	//	top left 
			-1.0f, -1.0f, -1.0f,//	bottom left 
			-1.0f, -1.0f, 1.0f,	//	bottom right

			//	Right face
			1.0f, 1.0f, -1.0f,	//	top right 
			1.0f, 1.0f, 1.0f,	//	top left 
			1.0f, -1.0f, 1.0f,	//	bottom left
			1.0f, -1.0f, -1.0f,	//	bottom Right 
		};
		
		float farrCubeNormals[] = new float[]
		{
			//	Top face
			0.0f, 1.0f, 0.0f,	//	top right 
			0.0f, 1.0f, 0.0f,	//	top left 
			0.0f, 1.0f, 0.0f,	//	bottom left
			0.0f, 1.0f, 0.0f,	//	bottom right 

			//	Bottom face
			0.0f, -1.0f, 0.0f,//	top right 
			0.0f, -1.0f, 0.0f,	//	top left 
			0.0f, -1.0f, 0.0f,	//	bottom left 
			0.0f, -1.0f, 0.0f,	//	bottom right

			//	Front face
			0.0f, 0.0f, 1.0f,	//	top right 
			0.0f, 0.0f, 1.0f,	//	top left 
			0.0f, 0.0f, 1.0f,	//	bottom left 
			0.0f, 0.0f, 1.0f,	//	bottom Right 

			//	Back face
			0.0f, 0.0f, -1.0f,	//	top Right 
			0.0f, 0.0f, -1.0f,//	top left 
			0.0f, 0.0f, -1.0f,	//	bottom left 
			0.0f, 0.0f, -1.0f,	//	bottom right 

			//	Left face
			-1.0f, 0.0f, 0.0f,	//	top right 
			-1.0f, 0.0f, 0.0f,	//	top left 
			-1.0f, 0.0f, 0.0f,//	bottom left 
			-1.0f, 0.0f, 0.0f,	//	bottom right

			//	Right face
			1.0f, 0.0f, 0.0f,	//	top right 
			1.0f, 0.0f, 0.0f,	//	top left 
			1.0f, 0.0f, 0.0f,	//	bottom left
			1.0f, 0.0f, 0.0f,	//	bottom Right 
		};
		
		ByteBuffer byBuffer;
		FloatBuffer fVerticesBuffer;

		///////////////////////////////////////////////////////////////////
		//+	VAO Triangle
		
		GLES30.glGenVertexArrays(1, VAO, 0);
		GLES30.glBindVertexArray(VAO[0]);
		
		///////////////////////////////////////////////////////////////////
		//+	VBO Position
		GLES30.glGenBuffers(1, VBOPosition, 0);
		GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, VBOPosition[0]);
		
		//	Convert array to byte data.
		byBuffer = ByteBuffer.allocateDirect(farrCubeVertices.length * 4);
		byBuffer.order(ByteOrder.nativeOrder());
		fVerticesBuffer = byBuffer.asFloatBuffer();
		fVerticesBuffer.put(farrCubeVertices);
		fVerticesBuffer.position(0);
		
		GLES30.glBufferData(
							GLES30.GL_ARRAY_BUFFER,
							farrCubeVertices.length * 4,
							fVerticesBuffer,
							GLES30.GL_STATIC_DRAW
							);
							
		GLES30.glVertexAttribPointer(GLESMacros.RTR_ATTRIBUTE_POSITION, 3, GLES30.GL_FLOAT, false, 0, 0);
		
		GLES30.glEnableVertexAttribArray(GLESMacros.RTR_ATTRIBUTE_POSITION);
		
		GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0);
		//-	VBO Position
		///////////////////////////////////////////////////////////////////
		
		///////////////////////////////////////////////////////////////////
		//+	VBO Normal
		GLES30.glGenBuffers(1, VBONormal, 0);
		GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, VBONormal[0]);
		
		//	Convert array to byte data.
		byBuffer = ByteBuffer.allocateDirect(farrCubeNormals.length * 4);
		byBuffer.order(ByteOrder.nativeOrder());
		fVerticesBuffer = byBuffer.asFloatBuffer();
		fVerticesBuffer.put(farrCubeNormals);
		fVerticesBuffer.position(0);
		
		GLES30.glBufferData(
							GLES30.GL_ARRAY_BUFFER,
							farrCubeNormals.length * 4,
							fVerticesBuffer,
							GLES30.GL_STATIC_DRAW
							);
							
		GLES30.glVertexAttribPointer(GLESMacros.RTR_ATTRIBUTE_NORMAL, 3, GLES30.GL_FLOAT, false, 0, 0);
		
		GLES30.glEnableVertexAttribArray(GLESMacros.RTR_ATTRIBUTE_NORMAL);
		
		GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0);
		//-	VBO Color
		///////////////////////////////////////////////////////////////////
		
		GLES30.glBindVertexArray(0);
		
		//-	VAO Triangle
		///////////////////////////////////////////////////////////////////
		
		//	Enable depth testing.
		GLES30.glEnable(GLES30.GL_DEPTH_TEST);
		//	dept test to do
		GLES30.glDepthFunc(GLES30.GL_LEQUAL);
		//	We will always cull backfaces for better performance
		GLES30.glEnable(GLES30.GL_CULL_FACE);
		
		//	Set the background frame color.
		GLES30.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		
		//	Set Projection matrix to identity.
		Matrix.setIdentityM(perspectiveProjectionMatrix, 0);
	}
	
	private void resize(int width, int height)
	{
		if (height == 0)
			height = 1;

		//	Adjust the viewport based on geometry changes, 
		//	such as screen rotation.
		GLES30.glViewport(0, 0, width, height);
		Matrix.perspectiveM(perspectiveProjectionMatrix, 0, 45.0f, (float)width/(float)height, 0.1f, 100.0f);
	}
	
	public void draw()
	{
		//	Draw background color.
		GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT | GLES30.GL_DEPTH_BUFFER_BIT);
		
		//	Use shader program.
		GLES30.glUseProgram(shaderObjectProgram);
		
		if (1 == g_iDoubleTap)
		{
			GLES30.glUniform1i(g_iDoubleTapUniform, 1);
			GLES30.glUniform3f(g_iLDVec3Uniform, 1.0f, 1.0f, 1.0f);	//	Diffuse
			GLES30.glUniform3f(g_iKDVec3Uniform, 0.5f, 0.5f, 0.5f);	//	grey effect

			float[] farrLightPosition = {0.0f, 0.0f, 2.0f, 1.0f};
			GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, farrLightPosition, 0);
		}
		else
		{
			GLES30.glUniform1i(g_iDoubleTapUniform, 0);
		}
		
		//	OpenGL-ES drawing.
		float modelMatrix[] = new float[16];
		float modelViewMatrix[] = new float[16];
		float rotationMatrix[] = new float[16];
		
		Matrix.setIdentityM(modelMatrix, 0);
		Matrix.setIdentityM(modelViewMatrix, 0);
		Matrix.setIdentityM(rotationMatrix, 0);
		
		Matrix.translateM(modelMatrix, 0, 0.0f,0.0f,-5.0f);
		Matrix.setRotateM(rotationMatrix, 0, g_fAngleCube, 1.0f, 1.0f, 1.0f);
		
		Matrix.multiplyMM(modelViewMatrix, 0, modelMatrix, 0, rotationMatrix, 0);
		
		GLES30.glUniformMatrix4fv(g_iModelViewMat4Uniform, 1, false, modelViewMatrix, 0);
		GLES30.glUniformMatrix4fv(g_iProjectionMat4Uniform, 1, false, perspectiveProjectionMatrix, 0);
		
		//	Bind VAO
		GLES30.glBindVertexArray(VAO[0]);
		
		GLES30.glDrawArrays(GLES30.GL_TRIANGLE_FAN, 0, 4);
		GLES30.glDrawArrays(GLES30.GL_TRIANGLE_FAN, 4, 4);
		GLES30.glDrawArrays(GLES30.GL_TRIANGLE_FAN, 8, 4);
		GLES30.glDrawArrays(GLES30.GL_TRIANGLE_FAN, 12, 4);
		GLES30.glDrawArrays(GLES30.GL_TRIANGLE_FAN, 16, 4);
		GLES30.glDrawArrays(GLES30.GL_TRIANGLE_FAN, 20, 4);
		
		// unbind VAO
		GLES30.glBindVertexArray(0);
		
		//	un-use program.
		GLES30.glUseProgram(0);
		
		//	Keep rotating
		if (g_iSingleTap == 1)
		{
			g_fAngleCube = g_fAngleCube - 0.75f;
		}
		
		//	render/flush
		requestRender();	//	Same a SwapBuffers()
	}
	
	void uninitialize()
	{
		System.out.println("RTR:Uninitialize-->");
		if (VAO[0] != 0)
		{
			GLES30.glDeleteVertexArrays(1, VAO, 0);
			VAO[0] = 0;
		}
		
		if (VBOPosition[0] != 0)
		{
			GLES30.glDeleteBuffers(1, VBOPosition, 0);
			VBOPosition[0] = 0;
		}
		
		if (VBONormal[0] != 0)
		{
			GLES30.glDeleteBuffers(1, VBONormal, 0);
			VBONormal[0] = 0;
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
