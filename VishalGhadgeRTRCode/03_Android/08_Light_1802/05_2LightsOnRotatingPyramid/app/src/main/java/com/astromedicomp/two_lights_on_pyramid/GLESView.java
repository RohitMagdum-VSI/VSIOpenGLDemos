package com.astromedicomp.two_lights_on_pyramid;

import android.content.Context;
import android.opengl.GLSurfaceView;//	for open gl surface view all related.
import javax.microedition.khronos.opengles.GL10;
import javax.microedition.khronos.egl.EGLConfig;
import android.opengl.GLES30;	//	Change this version as per requirement.
import android.view.MotionEvent;
import android.view.GestureDetector;
import android.view.GestureDetector.OnGestureListener;
import android.view.GestureDetector.OnDoubleTapListener;
import java.nio.ShortBuffer;

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

	//+	For pyramid
    private int[] vao_pyramid = new int[1];
    private int[] vbo_position = new int[1];
    private int[] vbo_normal = new int[1];
	//-	For pyramid

	//+ Uniforms
	private int g_iModelMat4Uniform;
	private int g_iViewMat4Uniform;
	private int g_iProjectionMat4Uniform;

	private int g_iRotationMatR4Uniform;
	private int g_iRotationMatG4Uniform;
	private int g_iRotationMatB4Uniform;

	//	Red Light / Right side light
	private int g_iLaRVec3Uniform;
	private int g_iLdRVec3Uniform;
	private int g_iLsRVec3Uniform;
	private int g_iLightPositionRVec4Uniform;

	//	Blue Light / Left side light
	private int g_iLaBVec3Uniform;
	private int g_iLdBVec3Uniform;
	private int g_iLsBVec3Uniform;
	private int g_iLightPositionBVec4Uniform;

	private int g_iKaVec3Uniform;
	private int g_iKdVec3Uniform;
	private int g_iKsVec3Uniform;
	private int g_iMaterialShininessUniform;

	private int g_iDoubleTapUniform;
	private int g_iSingleTapUniform;
	//- Uniforms

	//	Red Light
	private float g_farrLightRAmbient[] = { 0.0f, 0.0f, 0.0f, 0.0f };
	private float g_farrLightRDiffuse[] = { 1.0f, 0.0f, 0.0f, 0.0f };
	private float g_farrLightRSpecular[] = { 1.0f, 0.0f, 0.0f, 0.0f };
	private float g_farrLightRPosition[] = { 2.0f, 1.0f, 1.0f, 0.0f };	//	Decides position of light

	//	Blue Light
	private float g_farrLightBAmbient[] = { 0.0f, 0.0f, 0.0f, 0.0f };
	private float g_farrLightBDiffuse[] = { 0.0f, 0.0f, 1.0f, 0.0f };
	private float g_farrLightBSpecular[] = { 0.0f, 0.0f, 1.0f, 0.0f };
	private float g_farrLightBPosition[] = { -2.0f, 1.0f, 1.0f, 0.0f };	//	Decides position of light

	private float g_farrMaterialAmbient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	private float g_farrMaterialDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	private float g_farrMaterialSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	private float g_fMaterialShininess = 50.0f;
		
	private float perspectiveProjectionMatrix[] = new float[16];	//	4*4 Matrix.

    private float g_fAngle = 0.0f;
	private int g_iLightType = 0;	// For Vertex / Fragment
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

		if (0 != g_iDoubleTap) {
			g_iLightType++;
			if (g_iLightType > 1) {
				g_iLightType = 0;
			}
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
		"uniform mat4 u_model_matrix;"+
		"uniform mat4 u_view_matrix;"+
		"uniform mat4 u_projection_matrix;"+
		"uniform vec4 u_light_positionR;"+
		"uniform vec4 u_light_positionB;"+
		"uniform mediump int u_double_tap;"+
		"out vec3 transformed_normals;"+
		"out vec3 light_directionR;"+
		"out vec3 light_directionB;"+
		"out vec3 viewer_vector;"+
		"void main(void)"+
		"{"+
            "vec4 matNotUsed = u_model_matrix *  vPosition;"											+
            "mat4 matModelView = u_view_matrix;"											+
             "if (1 == u_double_tap)"+
			"{"+
                /*"vec4 eyeCoordinates = u_view_matrix * u_model_matrix * vPosition;"											+*/
                "vec4 eyeCoordinates = matModelView * vPosition;"											+
				"transformed_normals = normalize(mat3(matModelView) * vNormal);"											+
				"light_directionR = normalize(vec3(u_light_positionR) - eyeCoordinates.xyz);"											+
				"light_directionB = normalize(vec3(u_light_positionB) - eyeCoordinates.xyz);"											+
				"viewer_vector = normalize(-eyeCoordinates.xyz);"											+
			"}"+
			"gl_Position = u_projection_matrix * matModelView * vPosition;"+
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
		"in vec3 out_phong_ads_color;"				+
		"in vec3 transformed_normals;"			+
		"in vec3 light_directionR;"			+
		"in vec3 light_directionB;"			+
		"in vec3 viewer_vector;"			+
		"uniform vec3 u_LaR;	"				+
		"uniform vec3 u_LdR;	"				+
		"uniform vec3 u_LsR;	"				+
		"uniform vec3 u_LaB;	"				+
		"uniform vec3 u_LdB;	"				+
		"uniform vec3 u_LsB;	"				+
		"uniform vec3 u_Ka;"					+
		"uniform vec3 u_Kd;"					+
		"uniform vec3 u_Ks;"					+
		"uniform float u_material_shininess;"		+
		"uniform mediump int u_double_tap;"			+
		"out vec4 vFragColor;"						+
		"void main(void)"+
		"{"+
			"vec3 phong_ads_color;"+
			"if (1 == u_double_tap)"+
			"{"+
				"vec3 normalized_transformed_normals = normalize(transformed_normals);"											+
				"vec3 normalized_light_directionR = normalize(light_directionR);"											+
				"vec3 normalized_light_directionB = normalize(light_directionB);"											+
				"vec3 normalized_viewer_vector = normalize(viewer_vector);"											+
				/*Red Light*/
				"float tn_dot_ldR = max(dot(normalized_transformed_normals, normalized_light_directionR), 0.0);"											+
				"vec3 ambientR = u_LaR * u_Ka;"											+
				"vec3 diffuseR = u_LdR * u_Kd * tn_dot_ldR;"											+
				"vec3 reflection_vectorR = reflect(-normalized_light_directionR, normalized_transformed_normals);"											+
				"vec3 specularR = u_LsR * u_Ks * pow(max(dot(reflection_vectorR, normalized_viewer_vector), 0.0), u_material_shininess);"											+
				/*Blue Light*/
				"float tn_dot_ldB = max(dot(normalized_transformed_normals, normalized_light_directionB), 0.0);"											+
				"vec3 ambientB = u_LaB * u_Ka;"											+
				"vec3 diffuseB = u_LdB * u_Kd * tn_dot_ldB;"											+
				"vec3 reflection_vectorB = reflect(-normalized_light_directionB, normalized_transformed_normals);"											+
				"vec3 specularB = u_LsB * u_Ks * pow(max(dot(reflection_vectorB, normalized_viewer_vector), 0.0), u_material_shininess);"											+
				"phong_ads_color = ambientR + ambientB + diffuseR + diffuseB + specularR + specularB;"											+
			"}"+
			"else"+
			"{"+
				"phong_ads_color = vec3(1.0,1.0,1.0);"+
			"}"+
			"vFragColor = vec4(phong_ads_color, 1.0);"+
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
		g_iModelMat4Uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_model_matrix");
		g_iViewMat4Uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_view_matrix");
		g_iProjectionMat4Uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_projection_matrix");
		g_iDoubleTapUniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_double_tap");

		//	Red Light
		g_iLaRVec3Uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_LaR");
		g_iLdRVec3Uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_LdR");
		g_iLsRVec3Uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_LsR");
		g_iLightPositionRVec4Uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_light_positionR");

		//	Blue Light
		g_iLaBVec3Uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_LaB");
		g_iLdBVec3Uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_LdB");
		g_iLsBVec3Uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_LsB");
		g_iLightPositionBVec4Uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_light_positionB");

		g_iKaVec3Uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_Ka");
		g_iKdVec3Uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_Kd");
		g_iKsVec3Uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_Ks");
		g_iMaterialShininessUniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_material_shininess");

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

		final float farrPyramidNormals[] = new float[]
		{
				//	Front face
				0.0f, 0.447214f, 0.894427f,	//	apex
				0.0f, 0.447214f, 0.894427f,	//	left_bottom
				0.0f, 0.447214f, 0.894427f,	//	right_bottom

				//	Right face
				0.894427f, 0.447214f, 0.0f,	//	apex
				0.894427f, 0.447214f, 0.0f,	//	left_bottom
				0.894427f, 0.447214f, 0.0f,	//	right_bottom

				//	Back face
				0.0f, 0.447214f, -0.894427f,	//	apex
				0.0f, 0.447214f, -0.894427f,	//	left_bottom
				0.0f, 0.447214f, -0.894427f,	//	right_bottom

				//	Left face
				-0.894427f, 0.447214f, 0.0f,	//	apex
				-0.894427f, 0.447214f, 0.0f,	//	left_bottom
				-0.894427f, 0.447214f, 0.0f,	//	right_bottom
		};
		
		ByteBuffer byBuffer;
		FloatBuffer fVerticesBuffer;

		///////////////////////////////////////////////////////////////////
		//+	VAO Pyramid
		
		GLES30.glGenVertexArrays(1, vao_pyramid, 0);
		GLES30.glBindVertexArray(vao_pyramid[0]);
		
		///////////////////////////////////////////////////////////////////
		//+	VBO Position
		GLES30.glGenBuffers(1, vbo_position, 0);
		GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vbo_position[0]);
		
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
		//+	VBO Normal
		GLES30.glGenBuffers(1, vbo_normal, 0);
		GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vbo_normal[0]);
		
		//	Convert array to byte data.
		byBuffer = ByteBuffer.allocateDirect(farrPyramidNormals.length * 4);
		byBuffer.order(ByteOrder.nativeOrder());
		fVerticesBuffer = byBuffer.asFloatBuffer();
		fVerticesBuffer.put(farrPyramidNormals);
		fVerticesBuffer.position(0);
		
		GLES30.glBufferData(
							GLES30.GL_ARRAY_BUFFER,
						farrPyramidNormals.length * 4,
							fVerticesBuffer,
							GLES30.GL_STATIC_DRAW
							);
							
		GLES30.glVertexAttribPointer(GLESMacros.RTR_ATTRIBUTE_NORMAL, 3, GLES30.GL_FLOAT, false, 0, 0);
		
		GLES30.glEnableVertexAttribArray(GLESMacros.RTR_ATTRIBUTE_NORMAL);
		
		GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0);
		//-	VBO Normal
		///////////////////////////////////////////////////////////////////

		GLES30.glBindVertexArray(0);
		//-	VAO Pyramid
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

			GLES30.glUniform3fv(g_iLaRVec3Uniform, 1, g_farrLightRAmbient, 0);
			GLES30.glUniform3fv(g_iLdRVec3Uniform, 1, g_farrLightRDiffuse, 0);
			GLES30.glUniform3fv(g_iLsRVec3Uniform, 1, g_farrLightRSpecular, 0);
            GLES30.glUniform4fv(g_iLightPositionRVec4Uniform, 1, g_farrLightRPosition, 0);

			GLES30.glUniform3fv(g_iLaBVec3Uniform, 1, g_farrLightBAmbient, 0);
			GLES30.glUniform3fv(g_iLdBVec3Uniform, 1, g_farrLightBDiffuse, 0);
			GLES30.glUniform3fv(g_iLsBVec3Uniform, 1, g_farrLightBSpecular, 0);
            GLES30.glUniform4fv(g_iLightPositionBVec4Uniform, 1, g_farrLightBPosition, 0);

			GLES30.glUniform3fv(g_iKaVec3Uniform, 1, g_farrMaterialAmbient, 0);
			GLES30.glUniform3fv(g_iKdVec3Uniform, 1, g_farrMaterialDiffuse, 0);
			GLES30.glUniform3fv(g_iKsVec3Uniform, 1, g_farrMaterialSpecular, 0);
			GLES30.glUniform1f(g_iMaterialShininessUniform, g_fMaterialShininess);
		}
		else
		{
			GLES30.glUniform1i(g_iDoubleTapUniform, 0);
		}
		
		//	OpenGL-ES drawing.
		float modelMatrix[] = new float[16];
        float viewMatrix[] = new float[16];
        float rotationMatrix[] = new float[16];

		Matrix.setIdentityM(modelMatrix, 0);
        Matrix.setIdentityM(viewMatrix, 0);
        Matrix.setIdentityM(rotationMatrix, 0);

		Matrix.translateM(modelMatrix, 0, 0.0f,0.0f,-6.0f);
        Matrix.rotateM(rotationMatrix,0,g_fAngle,0.0f,1.0f,0.0f);    //  Y-axis rotation
        Matrix.multiplyMM(viewMatrix,0,modelMatrix,0,rotationMatrix,0);
		
		GLES30.glUniformMatrix4fv(g_iModelMat4Uniform, 1, false, modelMatrix, 0);
		GLES30.glUniformMatrix4fv(g_iViewMat4Uniform, 1, false, viewMatrix, 0);
		GLES30.glUniformMatrix4fv(g_iProjectionMat4Uniform, 1, false, perspectiveProjectionMatrix, 0);

        //	Bind VAO
		GLES30.glBindVertexArray(vao_pyramid[0]);
		
		GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 12);
		
		// unbind VAO
		GLES30.glBindVertexArray(0);
		
		//	un-use program.
		GLES30.glUseProgram(0);

		update();

		//	render/flush
		requestRender();	//	Same a SwapBuffers()
	}


    void update()
    {
        float fSpeed = 1.0f;

        g_fAngle = g_fAngle + fSpeed;
        if (g_fAngle >= 360)
        {
            g_fAngle = 0.0f;
        }
    }
	
	void uninitialize()
	{
		System.out.println("RTR:Uninitialize-->");
		if (vao_pyramid[0] != 0)
		{
			GLES30.glDeleteVertexArrays(1, vao_pyramid, 0);
            vao_pyramid[0] = 0;
		}
		
		if (vbo_position[0] != 0)
		{
			GLES30.glDeleteBuffers(1, vbo_position, 0);
            vbo_position[0] = 0;
		}
		
		if (vbo_normal[0] != 0)
		{
			GLES30.glDeleteBuffers(1, vbo_normal, 0);
            vbo_normal[0] = 0;
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
