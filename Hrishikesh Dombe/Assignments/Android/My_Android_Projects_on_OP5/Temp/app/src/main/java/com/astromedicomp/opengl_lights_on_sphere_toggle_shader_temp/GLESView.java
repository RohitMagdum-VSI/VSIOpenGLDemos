package com.astromedicomp.opengl_lights_on_sphere_toggle_shader_temp;

import android.content.Context; //For drawing context related
import android.opengl.GLSurfaceView; //For OpenGL Surface View and all related
import javax.microedition.khronos.opengles.GL10; //For OpenGL ES 1.0 needed as param type GL10
import javax.microedition.khronos.egl.EGLConfig; //For EGLConfig needed as param type EGLConfig
import android.opengl.GLES31; //For OpenGL ES 3.1
import android.view.MotionEvent; //For MotionEvent
import android.view.GestureDetector; //For GestureDetector
import android.view.GestureDetector.OnGestureListener; //For OnGestureListener
import android.view.GestureDetector.OnDoubleTapListener; //For OnDoubleTapListener

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import android.opengl.Matrix; //For Matrix Maths

import java.nio.ShortBuffer; //For Sphere.java

//A view for OpenGL ES3 Graphics which also receives touch events
public class GLESView extends GLSurfaceView implements GLSurfaceView.Renderer, OnGestureListener, OnDoubleTapListener
{
	private final Context context;

	private GestureDetector gestureDetector;

	private int vertexShaderObject_PVL;
	private int fragmentShaderObject_PVL;
	private int shaderProgramObject_PVL;

	private int vertexShaderObject_PFL;
	private int fragmentShaderObject_PFL;
	private int shaderProgramObject_PFL;

	//New, For Sphere.java
	private int numVertices;
	private int numElements;

	private int[] vao_sphere = new int[1];
	private int[] vbo_sphere_position = new int[1];
	private int[] vbo_sphere_normal = new int[1];
	private int[] vbo_sphere_element = new int[1];

	private float light_ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	private float light_diffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	private float light_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	private float light_position[] = { 100.0f, 100.0f, 100.0f, 1.0f };

	private float material_ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	private float material_diffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	private float material_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	private float material_shininess = 50.0f;

	private int modelMatrixUniform, viewMatrixUniform, projectionMatrixUniform;
	private int laUniform, ldUniform, lsUniform, lightPositionUniform;
	private int kaUniform, kdUniform, ksUniform, materialShininessUniform;

	private int modelMatrixUniform_PFL, viewMatrixUniform_PFL, projectionMatrixUniform_PFL;
	private int laUniform_PFL, ldUniform_PFL, lsUniform_PFL, lightPositionUniform_PFL;
	private int kaUniform_PFL, kdUniform_PFL, ksUniform_PFL, materialShininessUniform_PFL;

	private int doubleTapUniform, doubleTapUniform_PFL; //To toggle lighting on double tap

	private float perspectiveProjectionMatrix[] = new float[16]; //4x4 Matrix

	private int doubleTap; //For Lights
	private int singleTap; //For Toggle Between Per Vertex and Per Fragment

	public GLESView(Context drawingContext)
	{
		super(drawingContext);

		context = drawingContext;

		//Accordingly set EGLContext to current supported version of OpenGL ES
		setEGLContextClientVersion(3);

		//Set Renderer for drawing on the GLSurfaceView
		setRenderer(this);

		//Render the view only when there is a change in the drawing data
		setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);

		gestureDetector = new GestureDetector(context, this, null, false);
		gestureDetector.setOnDoubleTapListener(this);
	}

	@Override //Init Code
	public void onSurfaceCreated(GL10 gl, EGLConfig config)
	{
		//Get OpenGL ES Version
		String glesVersion = gl.glGetString(GL10.GL_VERSION);
		System.out.println("HAD: OpenGLES Version = "+glesVersion);

		//Get GLSL Version
		String glslVersion = gl.glGetString(GLES31.GL_SHADING_LANGUAGE_VERSION);
		System.out.println("HAD: GLSL Version = "+glslVersion);

		initialize(gl);
	}

	@Override //Change Size Code
	public void onSurfaceChanged(GL10 unused, int width, int height)
	{
		resize(width, height);
	}

	@Override //Rendering Code
	public void onDrawFrame(GL10 unused)
	{
		display();
	}

	//Handling 'onTouchEvent' is the most important, because it triggers all the gesture and tap events
	@Override
	public boolean onTouchEvent(MotionEvent e)
	{
		//Code
		int eventaction = e.getAction();
		
		if(!gestureDetector.onTouchEvent(e))
			super.onTouchEvent(e);

		return(true);
	}

	@Override
	public boolean onDoubleTap(MotionEvent e)
	{
		//Code
		System.out.println("HAD: Double Tap");
		doubleTap++;

		if(doubleTap > 1)
			doubleTap = 0;

		return(true);
	}

	@Override
	public boolean onDoubleTapEvent(MotionEvent e)
	{
		//Do not write any code here, because, it is already written in onDoubleTap
		return(true);
	}

	@Override
	public boolean onSingleTapConfirmed(MotionEvent e)
	{
		//Code
		System.out.println("HAD: Single Tap");
		singleTap++;

		if(singleTap > 2)
			singleTap = 1;
		
		return(true);
	}

	@Override
	public boolean onDown(MotionEvent e)
	{
		return(true);
	}

	@Override
	public boolean onFling(MotionEvent e1, MotionEvent e2, float velocityX, float velocityY)
	{
		return(true);
	}

	@Override
	public void onLongPress(MotionEvent e)
	{
	}

	@Override
	public boolean onScroll(MotionEvent e1, MotionEvent e2, float distanceX, float distanceY)
	{
		uninitialize();
		System.exit(0);
		return(true);
	}

	@Override
	public void onShowPress(MotionEvent e)
	{
	}

	@Override
	public boolean onSingleTapUp(MotionEvent e)
	{
		return(true);
	}

	private void initialize(GL10 gl)
	{
		//--------------------Vertex Shader for Per Vertex Lighting--------------------

		//Create Shader
		vertexShaderObject_PVL = GLES31.glCreateShader(GLES31.GL_VERTEX_SHADER);

		//Vertex Shader Source Code
		final String vertexShaderSourceCode_PVL = String.format
		(
			"#version 300 es" +
			"\n" +
			"in vec4 vPosition;" +
			"in vec3 vNormal;" +
			"uniform mat4 u_model_matrix;" +
			"uniform mat4 u_view_matrix;" +
			"uniform mat4 u_projection_matrix;" +
			"uniform int u_double_tap;" +
			"uniform vec3 u_La;" +
			"uniform vec3 u_Ld;" +
			"uniform vec3 u_Ls;" +
			"uniform vec4 u_light_position;" +
			"uniform vec3 u_Ka;" +
			"uniform vec3 u_Kd;" +
			"uniform vec3 u_Ks;" +
			"uniform float u_material_shininess;" +
			"out vec3 phong_ads_color;" +
			"void main(void)" +
			"{" +
			"if(u_double_tap == 1)" +
			"{" +
			"vec4 eyeCoordinates = u_view_matrix * u_model_matrix * vPosition;" +
			"vec3 transformed_normals = normalize(mat3(u_view_matrix * u_model_matrix) * vNormal);" +
			"vec3 light_direction = normalize(vec3(u_light_position) - eyeCoordinates.xyz);" +
			"float tn_dot_ld = max(dot(transformed_normals, light_direction), 0.0f);" +
			"vec3 ambient = u_La * u_Ka;" +
			"vec3 diffuse = u_Ld * u_Kd * tn_dot_ld;" +
			"vec3 reflection_vector = reflect(-light_direction, transformed_normals);" +
			"vec3 viewer_vector = normalize(-eyeCoordinates.xyz);" +
			"vec3 specular = u_Ls * u_Ks * pow(max(dot(reflection_vector, viewer_vector), 0.0f), u_material_shininess);" +
			"phong_ads_color = ambient + diffuse + specular;" +
			"}" +
			"else" +
			"{" +
			"phong_ads_color = vec3(1.0f, 1.0f, 1.0f);" +
			"}" +
			"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" +
			"}"
		);

		//Provide Source Code to Shader
		GLES31.glShaderSource(vertexShaderObject_PVL, vertexShaderSourceCode_PVL);

		//Compile Shader and Check for Errors
		GLES31.glCompileShader(vertexShaderObject_PVL);

		int[] iShaderCompiledStatus = new int[1];
		int[] iInfoLogLength = new int[1];
		String szInfoLog = null;
		GLES31.glGetShaderiv(vertexShaderObject_PVL, GLES31.GL_COMPILE_STATUS, iShaderCompiledStatus, 0);

		if(iShaderCompiledStatus[0] == GLES31.GL_FALSE)
		{
			GLES31.glGetShaderiv(vertexShaderObject_PVL, GLES31.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);

			if(iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES31.glGetShaderInfoLog(vertexShaderObject_PVL);
				System.out.println("HAD: Vertex Shader Compilation Log For Per Vertex Lighting = "+szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}

		//--------------------Fragment Shader For Per Vertex Lighting--------------------
		
		//Create Shader
		fragmentShaderObject_PVL = GLES31.glCreateShader(GLES31.GL_FRAGMENT_SHADER);

		//Fragment Shader Source Code
		final String fragmentShaderSourceCode_PVL = String.format 
		(
			"#version 300 es" +
			"\n" +
			"precision highp float;" +
			"in vec3 phong_ads_color;" +
			"out vec4 FragColor;" +
			"void main(void)" +
			"{" +
			"FragColor = vec4(phong_ads_color, 1.0f);" +
			"}"
		);

		//Provide Source Code to Shader
		GLES31.glShaderSource(fragmentShaderObject_PVL, fragmentShaderSourceCode_PVL);

		//Compile Shader and Check for Errors
		GLES31.glCompileShader(fragmentShaderObject_PVL);

		iShaderCompiledStatus[0] = 0; //re-initialize
		iInfoLogLength[0] = 0; //re-initialize
		szInfoLog  = null; //re-initialize

		GLES31.glGetShaderiv(fragmentShaderObject_PVL, GLES31.GL_COMPILE_STATUS, iShaderCompiledStatus, 0);

		if(iShaderCompiledStatus[0] == GLES31.GL_FALSE)
		{
			GLES31.glGetShaderiv(fragmentShaderObject_PVL, GLES31.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);

			if(iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES31.glGetShaderInfoLog(fragmentShaderObject_PVL);
				System.out.println("HAD: Fragment Shader Compilation Log For Per Vertex Lighting = "+szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}

		//--------------------Shader Program For Per Vertex Lighting--------------------

		//Create Shader Program
		shaderProgramObject_PVL = GLES31.glCreateProgram();

		//Attach Vertex Shader to Shader Program
		GLES31.glAttachShader(shaderProgramObject_PVL, vertexShaderObject_PVL);

		//Attach Fragment Shader to Shader Program
		GLES31.glAttachShader(shaderProgramObject_PVL, fragmentShaderObject_PVL);

		//Pre-link Binding of Shader Program Object with Vertex Shader Attributes
		GLES31.glBindAttribLocation(shaderProgramObject_PVL, GLESMacros.HAD_ATTRIBUTE_VERTEX, "vPosition");

		GLES31.glBindAttribLocation(shaderProgramObject_PVL, GLESMacros.HAD_ATTRIBUTE_NORMAL, "vNormal");

		//Link the Two Shaders together to the Shader Program Object
		GLES31.glLinkProgram(shaderProgramObject_PVL);
		
		int[] iShaderProgramLinkStatus = new int[1];
		iInfoLogLength[0] = 0; //re-initialize
		szInfoLog = null; //re-initialize

		GLES31.glGetProgramiv(shaderProgramObject_PVL, GLES31.GL_LINK_STATUS, iShaderProgramLinkStatus, 0);

		if(iShaderProgramLinkStatus[0] == GLES31.GL_FALSE)
		{
			GLES31.glGetProgramiv(shaderProgramObject_PVL, GLES31.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);

			if(iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES31.glGetProgramInfoLog(shaderProgramObject_PVL);
				System.out.println("HAD: Shader Program Link Log For Per Vertex Lighting = "+szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}

		//Get Uniform Locations
		modelMatrixUniform = GLES31.glGetUniformLocation(shaderProgramObject_PVL, "u_model_matrix");

		viewMatrixUniform = GLES31.glGetUniformLocation(shaderProgramObject_PVL, "u_view_matrix");
		
		projectionMatrixUniform = GLES31.glGetUniformLocation(shaderProgramObject_PVL, "u_projection_matrix");

		doubleTapUniform = GLES31.glGetUniformLocation(shaderProgramObject_PVL, "u_double_tap");

		laUniform = GLES31.glGetUniformLocation(shaderProgramObject_PVL, "u_La");
		ldUniform = GLES31.glGetUniformLocation(shaderProgramObject_PVL, "u_Ld");
		lsUniform = GLES31.glGetUniformLocation(shaderProgramObject_PVL, "u_Ls");

		lightPositionUniform = GLES31.glGetUniformLocation(shaderProgramObject_PVL, "u_light_position");

		kaUniform = GLES31.glGetUniformLocation(shaderProgramObject_PVL, "u_Ka");
		kdUniform = GLES31.glGetUniformLocation(shaderProgramObject_PVL, "u_Kd");
		ksUniform = GLES31.glGetUniformLocation(shaderProgramObject_PVL, "u_Ks");

		materialShininessUniform = GLES31.glGetUniformLocation(shaderProgramObject_PVL, "u_material_shininess");		
		
		//------------------------------------------------------------------------------------------------------------------------//
















		

		//--------------------Fragment Shader For Per Fragment Lighting--------------------
		
		//Create Shader
		vertexShaderObject_PFL = GLES31.glCreateShader(GLES31.GL_VERTEX_SHADER);

		//Vertex Shader Source Code
		final String vertexShaderSourceCode_PFL = String.format
		(
			"#version 300 es" +
			"\n" +
			"in vec4 vPosition;" +
			"in vec3 vNormal;" +
			"uniform mat4 u_model_matrix;" +
			"uniform mat4 u_view_matrix;" +
			"uniform mat4 u_projection_matrix;" +
			"uniform mediump int u_double_tap;" +
			"uniform vec4 u_light_position;" +
			"out vec3 transformed_normals;" +
			"out vec3 light_direction;" +
			"out vec3 viewer_vector;" +
			"void main(void)" +
			"{" +
			"if(u_double_tap == 1)" +
			"{" +
			"vec4 eyeCoordinates = u_view_matrix * u_model_matrix * vPosition;" +
			"transformed_normals = mat3(u_view_matrix * u_model_matrix) * vNormal;" +
			"light_direction = vec3(u_light_position) - eyeCoordinates.xyz;" +
			"viewer_vector = -eyeCoordinates.xyz;" +
			"}" +
			"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" +
			"}"
		);

		//Provide Source Code to Shader
		GLES31.glShaderSource(vertexShaderObject_PFL, vertexShaderSourceCode_PFL);

		//Compile Shader and Check for Errors
		GLES31.glCompileShader(vertexShaderObject_PFL);

		/*int[] iShaderCompiledStatus = new int[1];
		int[] iInfoLogLength = new int[1];
		String szInfoLog = null;
		*/

		iShaderCompiledStatus[0] = 0; //re-initialize
		iInfoLogLength[0] = 0; //re-initialize
		szInfoLog = null; //re-initialize

		GLES31.glGetShaderiv(vertexShaderObject_PFL, GLES31.GL_COMPILE_STATUS, iShaderCompiledStatus, 0);

		if(iShaderCompiledStatus[0] == GLES31.GL_FALSE)
		{
			GLES31.glGetShaderiv(vertexShaderObject_PFL, GLES31.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);

			if(iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES31.glGetShaderInfoLog(vertexShaderObject_PFL);
				System.out.println("HAD: Vertex Shader Compilation Log = "+szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}

		//--------------------Fragment Shader--------------------
		
		//Create Shader
		fragmentShaderObject_PFL = GLES31.glCreateShader(GLES31.GL_FRAGMENT_SHADER);

		//Fragment Shader Source Code
		final String fragmentShaderSourceCode_PFL = String.format 
		(
			"#version 300 es" +
			"\n" +
			"precision highp float;" +
			"in vec3 transformed_normals;" +
			"in vec3 light_direction;" +
			"in vec3 viewer_vector;" +
			"out vec4 FragColor;" +
			"uniform vec3 u_La;" +
			"uniform vec3 u_Ld;" +
			"uniform vec3 u_Ls;" +
			"uniform vec3 u_Ka;" +
			"uniform vec3 u_Kd;" +
			"uniform vec3 u_Ks;" +
			"uniform float u_material_shininess;" +
			"uniform int u_double_tap;" +
			"void main(void)" +
			"{" +
			"vec3 phong_ads_color;" +
			"if(u_double_tap == 1)" +
			"{" +
			"vec3 normalized_transformed_normals = normalize(transformed_normals);" +
			"vec3 normalized_light_direction = normalize(light_direction);" +
			"vec3 normalized_viewer_vector = normalize(viewer_vector);" +
			"vec3 ambient = u_La * u_Ka;" +
			"float tn_dot_ld = max(dot(normalized_transformed_normals, normalized_light_direction), 0.0f);" +
			"vec3 diffuse = u_Ld * u_Kd * tn_dot_ld;" +
			"vec3 reflection_vector = reflect(-normalized_light_direction, normalized_transformed_normals);" +
			"vec3 specular = u_Ls * u_Ks * pow(max(dot(reflection_vector, normalized_viewer_vector), 0.0f), u_material_shininess);" +
			"phong_ads_color = ambient + diffuse + specular;" +
			"}" +
			"else" +
			"{" +
			"phong_ads_color = vec3(1.0f, 1.0f, 1.0f);" +
			"}" +
			"FragColor = vec4(phong_ads_color, 1.0f);" +
			"}"
		);
		
		//Provide Source Codeto Shader
		GLES31.glShaderSource(fragmentShaderObject_PFL, fragmentShaderSourceCode_PFL);

		//Compile Shader and Check for Errors
		GLES31.glCompileShader(fragmentShaderObject_PFL);

		iShaderCompiledStatus[0] = 0; //re-initialize
		iInfoLogLength[0] = 0; //re-initialize
		szInfoLog  = null; //re-initialize

		GLES31.glGetShaderiv(fragmentShaderObject_PFL, GLES31.GL_COMPILE_STATUS, iShaderCompiledStatus, 0);

		if(iShaderCompiledStatus[0] == GLES31.GL_FALSE)
		{
			GLES31.glGetShaderiv(fragmentShaderObject_PFL, GLES31.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);

			if(iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES31.glGetShaderInfoLog(fragmentShaderObject_PFL);
				System.out.println("HAD: Fragment Shader Compilation Log = "+szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}

		//--------------------Shader Program--------------------

		//Create Shader Program
		shaderProgramObject_PFL = GLES31.glCreateProgram();

		//Attach Vertex Shader to Shader Program
		GLES31.glAttachShader(shaderProgramObject_PFL, vertexShaderObject_PFL);

		//Attach Fragment Shader to Shader Program
		GLES31.glAttachShader(shaderProgramObject_PFL, fragmentShaderObject_PFL);

		//Pre-link Binding of Shader Program Object with Vertex Shader Attributes
		GLES31.glBindAttribLocation(shaderProgramObject_PFL, GLESMacros.HAD_ATTRIBUTE_VERTEX, "vPosition");

		GLES31.glBindAttribLocation(shaderProgramObject_PFL, GLESMacros.HAD_ATTRIBUTE_NORMAL, "vNormal");

		//Link the Two Shaders together to the Shader Program Object
		GLES31.glLinkProgram(shaderProgramObject_PFL);
		
		//int[] iShaderProgramLinkStatus = new int[1];

		iShaderProgramLinkStatus[0] = 0;
		iInfoLogLength[0] = 0; //re-initialize
		szInfoLog = null; //re-initialize

		GLES31.glGetProgramiv(shaderProgramObject_PFL, GLES31.GL_LINK_STATUS, iShaderProgramLinkStatus, 0);

		if(iShaderProgramLinkStatus[0] == GLES31.GL_FALSE)
		{
			GLES31.glGetProgramiv(shaderProgramObject_PFL, GLES31.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);

			if(iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES31.glGetProgramInfoLog(shaderProgramObject_PFL);
				System.out.println("HAD: Shader Program Link Log = "+szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}

		//Get Uniform Locations
		modelMatrixUniform = GLES31.glGetUniformLocation(shaderProgramObject_PFL, "u_model_matrix");

		viewMatrixUniform = GLES31.glGetUniformLocation(shaderProgramObject_PFL, "u_view_matrix");
		
		projectionMatrixUniform = GLES31.glGetUniformLocation(shaderProgramObject_PFL, "u_projection_matrix");

		doubleTapUniform = GLES31.glGetUniformLocation(shaderProgramObject_PFL, "u_double_tap");

		laUniform = GLES31.glGetUniformLocation(shaderProgramObject_PFL, "u_La");
		ldUniform = GLES31.glGetUniformLocation(shaderProgramObject_PFL, "u_Ld");
		lsUniform = GLES31.glGetUniformLocation(shaderProgramObject_PFL, "u_Ls");

		lightPositionUniform = GLES31.glGetUniformLocation(shaderProgramObject_PFL, "u_light_position");

		kaUniform = GLES31.glGetUniformLocation(shaderProgramObject_PFL, "u_Ka");
		kdUniform = GLES31.glGetUniformLocation(shaderProgramObject_PFL, "u_Kd");
		ksUniform = GLES31.glGetUniformLocation(shaderProgramObject_PFL, "u_Ks");

		materialShininessUniform = GLES31.glGetUniformLocation(shaderProgramObject_PFL, "u_material_shininess");			
		//------------------------------------------------------------------------------------------------------------------------//
		

		//Vertices, Colors, Shader Attribs, Vao, Vbo Initializations
		Sphere sphere = new Sphere(); //Object of class Sphere

		float sphere_vertices[] = new float[1146];
		float sphere_normals[] = new float[1146];
		float sphere_textures[] = new float[764];
		short sphere_elements[] = new short[2280];

		sphere.getSphereVertexData(sphere_vertices, sphere_normals, sphere_textures, sphere_elements);

		numVertices = sphere.getNumberOfSphereVertices();
		numElements = sphere.getNumberOfSphereElements();

		//--------------------Vao for Sphere--------------------
		GLES31.glGenVertexArrays(1, vao_sphere, 0); //Taking Cassette
		GLES31.glBindVertexArray(vao_sphere[0]); //Binding with Vao
	
		//--------------------Vbo for Sphere Position--------------------
		GLES31.glGenBuffers(1, vbo_sphere_position, 0);
		GLES31.glBindBuffer(GLES31.GL_ARRAY_BUFFER, vbo_sphere_position[0]); //Binding with Vbo for Sphere Position

		//Byte Buffer
		ByteBuffer byteBuffer = ByteBuffer.allocateDirect(sphere_vertices.length * 4);
		byteBuffer.order(ByteOrder.nativeOrder());
		FloatBuffer verticesBuffer = byteBuffer.asFloatBuffer();
		verticesBuffer.put(sphere_vertices);
		verticesBuffer.position(0);

		GLES31.glBufferData(GLES31.GL_ARRAY_BUFFER, sphere_vertices.length * 4, verticesBuffer, GLES31.GL_STATIC_DRAW);
		
		GLES31.glVertexAttribPointer(GLESMacros.HAD_ATTRIBUTE_VERTEX, 3, GLES31.GL_FLOAT, false, 0, 0);

		GLES31.glEnableVertexAttribArray(GLESMacros.HAD_ATTRIBUTE_VERTEX);

		GLES31.glBindBuffer(GLES31.GL_ARRAY_BUFFER, 0); //Unbinding with Vabo for Sphere Position

		//--------------------Vbo for Sphere Normal--------------------
		GLES31.glGenBuffers(1, vbo_sphere_normal, 0);
		GLES31.glBindBuffer(GLES31.GL_ARRAY_BUFFER, vbo_sphere_normal[0]); //Binding with Vbo for Sphere Normal

		//Byte Buffer
		byteBuffer = ByteBuffer.allocateDirect(sphere_normals.length * 4);
		byteBuffer.order(ByteOrder.nativeOrder());
		verticesBuffer = byteBuffer.asFloatBuffer();
		verticesBuffer.put(sphere_normals);
		verticesBuffer.position(0);

		GLES31.glBufferData(GLES31.GL_ARRAY_BUFFER, sphere_normals.length * 4, verticesBuffer, GLES31.GL_STATIC_DRAW);

		GLES31.glVertexAttribPointer(GLESMacros.HAD_ATTRIBUTE_NORMAL, 3, GLES31.GL_FLOAT, false, 0, 0);
	
		GLES31.glEnableVertexAttribArray(GLESMacros.HAD_ATTRIBUTE_NORMAL);

		GLES31.glBindBuffer(GLES31.GL_ARRAY_BUFFER, 0); //Unbinding with Vbo for Sphere Normal

		//--------------------Vbo for Sphere Normal--------------------
		GLES31.glGenBuffers(1, vbo_sphere_element, 0);
		GLES31.glBindBuffer(GLES31.GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element[0]); //Binding with Vbo for Sphere Element

		//Byte Buffer
		byteBuffer = ByteBuffer.allocateDirect(sphere_elements.length * 2);
		byteBuffer.order(ByteOrder.nativeOrder());
		ShortBuffer elementsBuffer = byteBuffer.asShortBuffer();
		elementsBuffer.put(sphere_elements);
		elementsBuffer.position(0);

		GLES31.glBufferData(GLES31.GL_ELEMENT_ARRAY_BUFFER, sphere_elements.length * 2, elementsBuffer, GLES31.GL_STATIC_DRAW);

		GLES31.glBindBuffer(GLES31.GL_ELEMENT_ARRAY_BUFFER, 0); //Unbinding with Vbo for Sphere Element

		GLES31.glBindVertexArray(0); //Unbinding with Vao

		//Enable Depth Testing
		GLES31.glEnable(GLES31.GL_DEPTH_TEST);

		//Enable Depth Test to do
		GLES31.glDepthFunc(GLES31.GL_LEQUAL);

		//Cull back faces for better performance
		GLES31.glEnable(GLES31.GL_CULL_FACE);

		//Set Background Color
		GLES31.glClearColor(0.0f, 0.0f, 0.0f, 0.0f); //Black Color

		//initialization
		doubleTap = 0;
		singleTap = 1;

		//Set Projection Matrix to identity Matrix
		Matrix.setIdentityM(perspectiveProjectionMatrix, 0);
	}

	private void resize(int width, int height)
	{
		//Code
		GLES31.glViewport(0, 0, width, height);

		//Calculate the Perspective Projection
		Matrix.perspectiveM(perspectiveProjectionMatrix, 0, 45.0f, (float)width / (float)height, 0.1f, 100.0f);
	}

	public void display()
	{
		//Code
		GLES31.glClear(GLES31.GL_COLOR_BUFFER_BIT | GLES31.GL_DEPTH_BUFFER_BIT);

		//Use Shader Program
		if(singleTap == 1)
		{
			GLES31.glUseProgram(shaderProgramObject_PVL);
		
			/*if(doubleTap == 1)
			{
				GLES31.glUniform1i(doubleTapUniform, 1);

				//Setting Light Properties
				GLES31.glUniform3fv(laUniform, 1, light_ambient, 0);
				GLES31.glUniform3fv(ldUniform, 1, light_diffuse, 0);
				GLES31.glUniform3fv(lsUniform, 1, light_specular, 0);
				GLES31.glUniform4fv(lightPositionUniform, 1, light_position, 0);

				GLES31.glUniform3fv(kaUniform, 1, material_ambient, 0);
				GLES31.glUniform3fv(kdUniform, 1, material_diffuse, 0);
				GLES31.glUniform3fv(ksUniform, 1, material_specular, 0);
				GLES31.glUniform1f(materialShininessUniform, material_shininess);
			}
			else
			{
				GLES31.glUniform1i(doubleTapUniform, 0);
			}*/
		}
	
		else if(singleTap == 2)
		{
			GLES31.glUseProgram(shaderProgramObject_PFL);
		
			/*if(doubleTap == 1)
			{
				GLES31.glUniform1i(doubleTapUniform_PFL, 1);

				//Setting Light Properties
				GLES31.glUniform3fv(laUniform_PFL, 1, light_ambient, 0);
				GLES31.glUniform3fv(ldUniform_PFL, 1, light_diffuse, 0);
				GLES31.glUniform3fv(lsUniform_PFL, 1, light_specular, 0);
				GLES31.glUniform4fv(lightPositionUniform_PFL, 1, light_position, 0);

				GLES31.glUniform3fv(kaUniform_PFL, 1, material_ambient, 0);
				GLES31.glUniform3fv(kdUniform_PFL, 1, material_diffuse, 0);
				GLES31.glUniform3fv(ksUniform_PFL, 1, material_specular, 0);
				GLES31.glUniform1f(materialShininessUniform_PFL, material_shininess);
			}
			else
			{
				GLES31.glUniform1i(doubleTapUniform_PFL, 0);
			}*/
		}

		if(doubleTap == 1)
			{
				GLES31.glUniform1i(doubleTapUniform, 1);

				//Setting Light Properties
				GLES31.glUniform3fv(laUniform, 1, light_ambient, 0);
				GLES31.glUniform3fv(ldUniform, 1, light_diffuse, 0);
				GLES31.glUniform3fv(lsUniform, 1, light_specular, 0);
				GLES31.glUniform4fv(lightPositionUniform, 1, light_position, 0);

				GLES31.glUniform3fv(kaUniform, 1, material_ambient, 0);
				GLES31.glUniform3fv(kdUniform, 1, material_diffuse, 0);
				GLES31.glUniform3fv(ksUniform, 1, material_specular, 0);
				GLES31.glUniform1f(materialShininessUniform, material_shininess);
			}
			else
			{
				GLES31.glUniform1i(doubleTapUniform, 0);
			}

		//OpenGL ES Drawing
		float modelMatrix[] = new float[16];
		float viewMatrix[] = new float[16];

		//Set modelMatrix and viewMatrix to identity
		Matrix.setIdentityM(modelMatrix, 0);
		Matrix.setIdentityM(viewMatrix, 0);
				
		//Translate Z-Axis by -2.0f
		Matrix.translateM(modelMatrix, 0, 0.0f, 0.0f, -1.5f);

		//Pass above Model Matrix to the Vertex Shader in "u_model_matrix" Shader Variable
		GLES31.glUniformMatrix4fv(modelMatrixUniform, 1, false, modelMatrix, 0);

		//Pass above View Matrix to the Vertex Shader in "u_view_matrix" Shader Variable
		GLES31.glUniformMatrix4fv(viewMatrixUniform, 1, false, viewMatrix, 0);

		//Pass above Projection Matrix to the Vertex Shader in "u_projection_matrix" Shader Variable
		GLES31.glUniformMatrix4fv(projectionMatrixUniform, 1, false, perspectiveProjectionMatrix, 0);

		//Bind vao
		GLES31.glBindVertexArray(vao_sphere[0]);

		//Draw
		GLES31.glBindBuffer(GLES31.GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element[0]);
		GLES31.glDrawElements(GLES31.GL_TRIANGLES, numElements, GLES31.GL_UNSIGNED_SHORT, 0);

		//Unbind vao
		GLES31.glBindVertexArray(0);

		//Un-use Shader program
		GLES31.glUseProgram(0);

		//Render
		requestRender();
	}

	void uninitialize()
	{
		//Code
		//Destroy vao
		if(vao_sphere[0] != 0)
		{
			GLES31.glDeleteVertexArrays(1, vao_sphere, 0);
			vao_sphere[0] = 0;
		}

		//Destroy vbo_sphere_position
		if(vbo_sphere_position[0] != 0)
		{
			GLES31.glDeleteBuffers(1, vbo_sphere_position, 0);
			vbo_sphere_position[0] = 0;
		}

		//Destroy vbo_sphere_normal
		if(vbo_sphere_normal[0] != 0)
		{
			GLES31.glDeleteBuffers(1, vbo_sphere_normal, 0);
			vbo_sphere_normal[0] = 0;
		}

		//Destroy vbo_sphere_element
		if(vbo_sphere_element[0] != 0)
		{
			GLES31.glDeleteBuffers(1, vbo_sphere_element, 0);
			vbo_sphere_element[0] = 0;
		}

		if(shaderProgramObject_PVL != 0)
		{
			if(vertexShaderObject_PVL != 0)
			{
				//Detach Vertex Shader from Shader Program Object For Per Vertex Lighting
				GLES31.glDetachShader(shaderProgramObject_PVL, vertexShaderObject_PVL);

				//Delete Vertex Shader Object For Per Vertex Lighting
				GLES31.glDeleteShader(vertexShaderObject_PVL);

				vertexShaderObject_PVL = 0;
			}

			if(fragmentShaderObject_PVL != 0)
			{
				//Detach Fragment Shader from Shader Program Object For Per Vertex Lighting
				GLES31.glDetachShader(shaderProgramObject_PVL, fragmentShaderObject_PVL);
				 
				//Delete Fragment Shader Object For Per Vertex Lighting
				GLES31.glDeleteShader(fragmentShaderObject_PVL);

				fragmentShaderObject_PVL = 0;
			}
		}

		//Delete Shader Program Object For Per Vertex Lighting
		if(shaderProgramObject_PVL != 0)
		{
			GLES31.glDeleteProgram(shaderProgramObject_PVL);
			shaderProgramObject_PVL = 0;
		}

		if(shaderProgramObject_PFL != 0)
		{
			if(vertexShaderObject_PFL != 0)
			{
				//Detach Vertex Shader from Shader Program Object For Per Fragment Lighting
				GLES31.glDetachShader(shaderProgramObject_PFL, vertexShaderObject_PFL);

				//Delete Vertex Shader Object For Per Fragment Lighting
				GLES31.glDeleteShader(vertexShaderObject_PFL);

				vertexShaderObject_PFL = 0;
			}

			if(fragmentShaderObject_PFL != 0)
			{
				//Detach Fragment Shader from Shader Program Object For Per Fragment Lighting
				GLES31.glDetachShader(shaderProgramObject_PFL, fragmentShaderObject_PFL);

				//Delete Fragment Shader Object For Per Fragment Lighting
				GLES31.glDeleteShader(fragmentShaderObject_PFL);

				fragmentShaderObject_PFL = 0;
			}
		}

		//Delete Shader Program Object For Per Fragment Lighting
		if(shaderProgramObject_PFL != 0)
		{
			GLES31.glDeleteProgram(shaderProgramObject_PFL);
			shaderProgramObject_PFL = 0;
		}
	}
}
