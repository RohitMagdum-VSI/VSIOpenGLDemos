package com.astromedicomp.opengl_24_sphere_with_lights;

import android.content.Context; //For drawing context related
import android.opengl.GLSurfaceView; //For OpenGL Surface View and all related
import javax.microedition.khronos.opengles.GL10; //For OpenGLES 1.0 needed as param type GL10
import javax.microedition.khronos.egl.EGLConfig; //For EGLConfig needed as param type EGLConfig
import android.opengl.GLES32; // For OpenGLES 3.2
import android.view.Gravity;
import android.view.MotionEvent; // For "MotionEvent"
import android.view.GestureDetector; // For GestureDetector
import android.view.GestureDetector.OnGestureListener; // OnGestureListener
import android.view.GestureDetector.OnDoubleTapListener; // OnDoubleTapListener

//For vbo
import java.nio.ByteBuffer;//nio For non-blocking I/O
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;

//For math
import java.math.*;

//For Matrix math
import android.opengl.Matrix;

//A View for OpenGLES3 graphics which also receives touch events
public class GLESView extends GLSurfaceView implements GLSurfaceView.Renderer, OnGestureListener, OnDoubleTapListener
{
	//Class variables or class fields
	private final Context context;

	private GestureDetector gestureDetector;

	//Shader Object
	private int vertexShaderObject;
	private int fragmentShaderObject;

	//Program Object
	private int shaderProgramObject;

	//As there are no pointers in java, when we want to use any variable as out parameter
	//we use array with only one member.
	//Vertex Array Object
	private int[] vao_sphere = new int[1];
	private int[] vbo_position = new int[1];
	private int[] vbo_normal = new int[1];
	private int[] vbo_elements = new int[1];

	private double angleSphere;

	private int modelMatrixUniform, viewMatrixUniform, projectionMatrixUniform;
	private int laUniform_White,ldUniform_White,lsUnifrom_White,lightPositionUniform_White;
	private int kaUniform,kdUniform,ksUniform,materialShininessUniform;

	private float light_ambient_White[] = {0.0f,0.0f,0.0f,1.0f};
	private float light_diffuse_White[] = {1.0f,1.0f,1.0f,1.0f};
	private float light_specular_White[] = {1.0f,1.0f,1.0f,1.0f};
	private float light_position_White[] = {0.0f,0.0f,100.0f,0.0f};

	private float material_ambient[] = {0.0f,0.0f,0.0f,1.0f};
	private float material_diffuse[] = {1.0f,1.0f,1.0f,1.0f};
	private float material_specular[] = {1.0f,1.0f,1.0f,1.0f};
	private float material_shininess = 50.0f;

	private float material_ambient_1[] = { 0.0215f,0.1745f,0.0215f,1.0f };
	private float material_diffuse_1[] = { 0.07568f,0.61424f,0.07568f,1.0f };
	private float material_specular_1[] = { 0.633f,0.727811f,0.633f,1.0f };
	private float material_shininess_1 = 0.6f * 128.0f;

	private float material_ambient_2[] = { 0.135f,0.2225f,0.1575f,1.0f };
	private float material_diffuse_2[] = { 0.54f,0.89f,0.63f,1.0f };
	private float material_specular_2[] = { 0.316228f,0.316228f,0.316228f,1.0f };
	private float material_shininess_2 = 0.1f * 128.0f;

	private float material_ambient_3[] = { 0.05375f,0.05f,0.06625f,1.0f };
	private float material_diffuse_3[] = { 0.18275f,0.17f,0.22525f,1.0f };
	private float material_specular_3[] = { 0.332741f,0.328634f,0.346435f,1.0f };
	private float material_shininess_3 = 0.3f * 128.0f;

	private float material_ambient_4[] = { 0.25f,0.20725f,0.20725f,1.0f };
	private float material_diffuse_4[] = { 1.0f,0.829f,0.829f,1.0f };
	private float material_specular_4[] = { 0.296648f,0.296648f,0.296648f,1.0f };
	private float material_shininess_4 = 0.088f * 128.0f;

	private float material_ambient_5[] = { 0.1745f,0.01175f,0.01175f,1.0f };
	private float material_diffuse_5[] = { 0.61424f,0.04136f,0.04136f,1.0f };
	private float material_specular_5[] = { 0.727811f,0.626959f,0.626959f,1.0f };
	private float material_shininess_5 = 0.6f * 128.0f;

	private float material_ambient_6[] = { 0.1f,0.18725f,0.1745f,1.0f };
	private float material_diffuse_6[] = { 0.396f,0.74151f,0.69102f,1.0f };
	private float material_specular_6[] = { 0.297254f,0.30829f,0.306678f,1.0f };
	private float material_shininess_6 = 0.1f * 128.0f;

	private float material_ambient_7[] = { 0.329412f,0.223529f,0.027451f,1.0f };
	private float material_diffuse_7[] = { 0.780392f,0.568627f,0.113725f,1.0f };
	private float material_specular_7[] = { 0.992157f,0.941176f,0.807843f,1.0f };
	private float material_shininess_7 = 0.21794872f * 128.0f;

	private float material_ambient_8[] = { 0.2125f,0.1275f,0.054f,1.0f };
	private float material_diffuse_8[] = { 0.714f,0.4284f,0.18144f,1.0f };
	private float material_specular_8[] = { 0.393548f,0.271906f,0.166721f,1.0f };
	private float material_shininess_8 = 0.2f * 128.0f;

	private float material_ambient_9[] = { 0.25f,0.25f,0.25f,1.0f };
	private float material_diffuse_9[] = { 0.4f,0.4f,0.4f,1.0f };
	private float material_specular_9[] = { 0.774597f,0.774597f,0.774597f,1.0f };
	private float material_shininess_9 = 0.6f * 128.0f;

	private float material_ambient_10[] = { 0.19125f,0.0735f,0.0225f,1.0f };
	private float material_diffuse_10[] = { 0.7038f,0.27048f,0.0828f,1.0f };
	private float material_specular_10[] = { 0.256777f,0.137622f,0.086014f,1.0f };
	private float material_shininess_10 = 0.1f * 128.0f;

	private float material_ambient_11[] = { 0.24725f,0.1995f,0.0745f,1.0f };
	private float material_diffuse_11[] = { 0.75164f,0.60648f,0.22648f,1.0f };
	private float material_specular_11[] = { 0.628281f,0.555802f,0.366065f,1.0f };
	private float material_shininess_11 = 0.4f * 128.0f;

	private float material_ambient_12[] = { 0.19225f,0.19225f,0.19225f,1.0f };
	private float material_diffuse_12[] = { 0.50754f,0.50754f,0.50754f,1.0f };
	private float material_specular_12[] = { 0.508273f,0.508273f,0.508273f,1.0f };
	private float material_shininess_12 = 0.4f * 128.0f;

	private float material_ambient_13[] = { 0.0f,0.0f,0.0f,1.0f };
	private float material_diffuse_13[] = { 0.01f,0.01f,0.01f,1.0f };
	private float material_specular_13[] = { 0.5f,0.5f,0.5f,1.0f };
	private float material_shininess_13 = 0.25f * 128.0f;

	private float material_ambient_14[] = { 0.0f,0.1f,0.06f,1.0f };
	private float material_diffuse_14[] = { 0.0f,0.50980392f,0.50980392f,1.0f };
	private float material_specular_14[] = { 0.50196078f,0.50196078f,0.50196078f,1.0f };
	private float material_shininess_14 = 0.25f * 128.0f;

	private float material_ambient_15[] = { 0.0f,0.0f,0.0f,1.0f };
	private float material_diffuse_15[] = { 0.1f,0.35f,0.1f,1.0f };
	private float material_specular_15[] = { 0.45f,0.55f,0.45f,1.0f };
	private float material_shininess_15 = 0.25f * 128.0f;

	private float material_ambient_16[] = { 0.0f,0.0f,0.0f,1.0f };
	private float material_diffuse_16[] = { 0.5f,0.0f,0.0f,1.0f };
	private float material_specular_16[] = { 0.7f,0.6f,0.6f,1.0f };
	private float material_shininess_16 = 0.25f * 128.0f;

	private float material_ambient_17[] = { 0.0f,0.0f,0.0f,1.0f };
	private float material_diffuse_17[] = { 0.55f,0.55f,0.55f,1.0f };
	private float material_specular_17[] = { 0.70f,0.70f,0.70f,1.0f };
	private float material_shininess_17 = 0.25f * 128.0f;

	private float material_ambient_18[] = { 0.0f,0.0f,0.0f,1.0f };
	private float material_diffuse_18[] = { 0.5f,0.5f,0.0f,1.0f };
	private float material_specular_18[] = { 0.6f,0.6f,0.5f,1.0f };
	private float material_shininess_18 = 0.25f * 128.0f;

	private float material_ambient_19[] = { 0.02f,0.02f,0.02f,1.0f };
	private float material_diffuse_19[] = { 0.1f,0.1f,0.1f,1.0f };
	private float material_specular_19[] = { 0.4f,0.4f,0.4f,1.0f };
	private float material_shininess_19 = 0.078125f * 128.0f;

	private float material_ambient_20[] = { 0.0f,0.05f,0.05f,1.0f };
	private float material_diffuse_20[] = { 0.4f,0.5f,0.5f,1.0f };
	private float material_specular_20[] = { 0.04f,0.7f,0.7f,1.0f };
	private float material_shininess_20 = 0.078125f * 128.0f;

	private float material_ambient_21[] = { 0.0f,0.05f,0.0f,1.0f };
	private float material_diffuse_21[] = { 0.4f,0.5f,0.4f,1.0f };
	private float material_specular_21[] = { 0.04f,0.7f,0.04f,1.0f };
	private float material_shininess_21 = 0.078125f * 128.0f;

	private float material_ambient_22[] = { 0.05f,0.0f,0.0f,1.0f };
	private float material_diffuse_22[] = { 0.5f,0.4f,0.4f,1.0f };
	private float material_specular_22[] = { 0.7f,0.04f,0.04f,1.0f };
	private float material_shininess_22 = 0.078125f * 128.0f;

	private float material_ambient_23[] = { 0.05f,0.05f,0.05f,1.0f };
	private float material_diffuse_23[] = { 0.5f,0.5f,0.5f,1.0f };
	private float material_specular_23[] = { 0.7f,0.7f,0.7f,1.0f };
	private float material_shininess_23 = 0.078125f * 128.0f;

	private float material_ambient_24[] = { 0.05f,0.05f,0.0f,1.0f };
	private float material_diffuse_24[] = { 0.5f,0.5f,0.4f,1.0f };
	private float material_specular_24[] = { 0.7f,0.7f,0.04f,1.0f };
	private float material_shininess_24 = 0.078125f * 128.0f;

	private float modelMatrix[] = new float[16];
	private float viewMatrix[] = new float[16];
	private float scaleMatrix[] = new float[16];

	private int doubleTapUniform;

	//4 x 4 matrix
	private float perspectiveProjectionMatrix[] = new float[16];

	private int doubleTap,singleTap,longPress;

	Sphere sphere = new Sphere();
	float sphere_vertices[] = new float[1146];
	float sphere_normals[] = new float[1146];
	float sphere_textures[] = new float[764];
	short sphere_elements[] = new short[2280];

	int iNumVertices,iNumElements;

	public GLESView(Context drawingContext)
	{
		super(drawingContext);

		context = drawingContext;

		//Accordingly set EGLContext to current supported version of OpenGL-ES
		setEGLContextClientVersion(3);

		//Set Renderer for drawing on GLSurfaceView
		setRenderer(this);

		//Render the view only when there is a change in drawing data
		setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);

		//this means 'handle' i.e who is going to handle 
		gestureDetector = new GestureDetector(context,this,null,false);
	
		//this means 'handle' i.e who is going to handle
		gestureDetector.setOnDoubleTapListener(this);
	}

	//overriden method of GLSurfaceView.Renderer(Init Code)
	@Override
	public void onSurfaceCreated(GL10 gl,EGLConfig config)
	{
		//OpenGL-ES version check
		String version = gl.glGetString(GL10.GL_VERSION);
		System.out.println("HAD: OpenGL-ES Version"+version);//"+" for concatination

		String glslVersion=gl.glGetString(GLES32.GL_SHADING_LANGUAGE_VERSION);
		System.out.println("HAD: GLSL Version"+glslVersion);

		initialize(gl);
	}

	//overriden method of GLSurfaceView.Renderer(Change Size Code)
	@Override
	public void onSurfaceChanged(GL10 unused,int width,int height)
	{
		resize(width,height);
	}

	//overriden method of GLSurfaceView.Renderer(Rendering Code)
	@Override
	public void onDrawFrame(GL10 unused)
	{
		draw();
	}

	//Handling 'onTouchEvent' Is The Most IMPORTANT, Because It Triggers All Gesture And Events
	@Override
	public boolean onTouchEvent(MotionEvent e)
	{
		//code
		int eventaction=e.getAction();
		if(!gestureDetector.onTouchEvent(e))
			super.onTouchEvent(e);
		return(true);
	}

	//abstract method from OnDoubleTapListener so must be implemented
	@Override
	public boolean onDoubleTap(MotionEvent e)
	{
		doubleTap++;
		if(doubleTap>1)
			doubleTap=0;
		return(true);
	}

	//abstract method from OnDoubleTapListener so must be implemented
	@Override
	public boolean onDoubleTapEvent(MotionEvent e)
	{
		//Do not write any code here because already written 'onDOubleTap'
		return(true);
	}

	//abstract method from OnDoubleTapListener so must be implemented
	@Override
	public boolean onSingleTapConfirmed(MotionEvent e)
	{
		singleTap++;
		if(singleTap>3)
			singleTap=0;
		return(true);
	}

	//abstract method from OnGestureListener so must be implemented
	@Override
	public boolean onDown(MotionEvent e)
	{
		//Do not write any code here because already written 'onSingleTapConfirmed'
		return(true);
	}

	//abstract method from OnGestureListener so must be implemented
	@Override
	public boolean onFling(MotionEvent e1,MotionEvent e2,float velocityX,float velocityY)
	{
		return(true);
	}

	//abstract method from OnGestureListener so must be implemented
	@Override
	public void onLongPress(MotionEvent e)
	{
		longPress++;
		if(longPress>1)
		longPress = 0;
	}

	//abstract method from OnGestureListener so must be implemented
	@Override
	public boolean onScroll(MotionEvent e1,MotionEvent e2,float distanceX,float distanceY)
	{
		uninitialize();
		System.exit(0);
		return(true);
	}

	//abstract method from OnGestureListener so must be implemented
	@Override
	public void onShowPress(MotionEvent e)
	{

	}

	//abstract method from OnGestureListener so must be implemented
	@Override
	public boolean onSingleTapUp(MotionEvent e)
	{
		return(true);
	}

	private void initialize(GL10 gl)
	{
		//Vertex Shader
		//Create Vertex Shader

		vertexShaderObject=GLES32.glCreateShader(GLES32.GL_VERTEX_SHADER);

		//vertex shader source code
		final String vertexShaderSourceCode = String.format
		(
			"#version 320 es"+
			"\n"+
			"in vec4 vPosition;"+
			"in vec3 vNormal;"+
			"uniform mat4 u_model_matrix;"+
			"uniform mat4 u_view_matrix;"+
			"uniform mat4 u_projection_matrix;"+
			"uniform mediump int u_double_tap;"+
			"uniform vec4 u_light_position_white;"+
			"out vec3 transformed_normals;"+
			"out vec3 light_direction_white;"+
			"out vec3 viewer_vector;"+
			"void main(void)"+
			"{"+
			"if(u_double_tap == 1)"+
			"{"+
			"vec4 eye_coordinates = u_view_matrix * u_model_matrix * vPosition;"+
			"transformed_normals = mat3(u_view_matrix * u_model_matrix) * vNormal;"+
			"light_direction_white = vec3(u_light_position_white) - eye_coordinates.xyz;"+
			"viewer_vector = -eye_coordinates.xyz;"+
			"}"+
			"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;"+
			"}"
		);

		//Provide Source Code to Shader
		GLES32.glShaderSource(vertexShaderObject,vertexShaderSourceCode);

		//Compile Shader & Check errors
		GLES32.glCompileShader(vertexShaderObject);
		int[] iShaderCompiledStatus = new int[1];
		int[] iInfoLogLength = new int[1];
		String szInfoLog = null; 
		GLES32.glGetShaderiv(vertexShaderObject,GLES32.GL_COMPILE_STATUS,iShaderCompiledStatus,0);
		if(iShaderCompiledStatus[0] == GLES32.GL_FALSE)
		{
			GLES32.glGetShaderiv(vertexShaderObject,GLES32.GL_INFO_LOG_LENGTH,iInfoLogLength,0);
			if(iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES32.glGetShaderInfoLog(vertexShaderObject);
				System.out.println("HAD: Vertex Shader Compilation Log = "+szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}

		//Fragment Shader
		//Create Shader
		fragmentShaderObject = GLES32.glCreateShader(GLES32.GL_FRAGMENT_SHADER);

		//Fragment Shader Source Code
		final String fragmentShaderSourceCode = String.format
		(
			"#version 320 es"+
			"\n"+
			"precision highp float;"+
			"in vec3 transformed_normals;"+
			"in vec3 light_direction_white;"+
			"in vec3 viewer_vector;"+
			"out vec4 FragColor;"+
			"uniform vec3 u_La_white;"+
			"uniform vec3 u_Ld_white;"+
			"uniform vec3 u_Ls_white;"+
			"uniform vec3 u_Ka;"+
			"uniform vec3 u_Kd;"+
			"uniform vec3 u_Ks;"+
			"uniform float u_material_shininess;"+
			"uniform int u_double_tap;"+
			"void main(void)"+
			"{"+
			"vec3 phong_ads_color;"+
			"if(u_double_tap == 1)"+
			"{"+
			"vec3 normalized_transformed_normals = normalize(transformed_normals);"+
			"vec3 normalized_light_direction_white = normalize(light_direction_white);"+
			"vec3 normalized_viewer_vector = normalize(viewer_vector);"+
			"vec3 ambient = u_La_white * u_Ka;"+
			"float tn_dot_ld_white = max(dot(normalized_transformed_normals,normalized_light_direction_white),0.0);"+
			"vec3 diffuse = u_Ld_white * u_Kd * tn_dot_ld_white;"+
			"vec3 reflection_vector_white = reflect(-normalized_light_direction_white,normalized_transformed_normals);"+
			"vec3 specular = u_Ls_white * u_Ks * pow(max(dot(reflection_vector_white,normalized_viewer_vector),0.0),u_material_shininess);"+
			"phong_ads_color=ambient+diffuse+specular;"+
			"}"+
			"else"+
			"{"+
			"phong_ads_color = vec3(1.0,1.0,1.0);"+
			"}"+
			"FragColor = vec4(phong_ads_color,1.0);"+
			"}"
		);

		GLES32.glShaderSource(fragmentShaderObject,fragmentShaderSourceCode);

		//Compile Shader and check for errors
		GLES32.glCompileShader(fragmentShaderObject);
		iShaderCompiledStatus[0] = 0;//reinitialize
		iInfoLogLength[0] = 0;//reinitialize
		szInfoLog=null;//reinitialize

		GLES32.glGetShaderiv(fragmentShaderObject,GLES32.GL_COMPILE_STATUS,iShaderCompiledStatus,0);
		if(iShaderCompiledStatus[0] == GLES32.GL_FALSE)
		{
			GLES32.glGetShaderiv(fragmentShaderObject,GLES32.GL_INFO_LOG_LENGTH,iInfoLogLength,0);
			if(iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES32.glGetShaderInfoLog(fragmentShaderObject);
				System.out.println("HAD: Fragment Shader Compilation Log = "+szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}

		//Create Shader Program
		shaderProgramObject = GLES32.glCreateProgram();

		//Attach Vertex Shader to Program Object
		GLES32.glAttachShader(shaderProgramObject,vertexShaderObject);

		//Attach Fragment Shader to Program Object
		GLES32.glAttachShader(shaderProgramObject,fragmentShaderObject);

		//Pre-link binding of shader program object with vertex shader attributes
		GLES32.glBindAttribLocation(shaderProgramObject,GLESMacros.HAD_ATTRIBUTE_VERTEX,"vPosition");
		GLES32.glBindAttribLocation(shaderProgramObject,GLESMacros.HAD_ATTRIBUTE_NORMAL,"vNormal");

		//Link 2 shaders together to shader program object
		GLES32.glLinkProgram(shaderProgramObject);
		int[] iShaderProgramLinkStatus = new int[1];
		iInfoLogLength[0] = 0;//reinitialize
		szInfoLog = null;//reinitialize

		GLES32.glGetProgramiv(shaderProgramObject,GLES32.GL_LINK_STATUS,iShaderProgramLinkStatus,0);
		if(iShaderProgramLinkStatus[0] == GLES32.GL_FALSE)
		{
			GLES32.glGetProgramiv(shaderProgramObject,GLES32.GL_INFO_LOG_LENGTH,iInfoLogLength,0);
			if(iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES32.glGetProgramInfoLog(shaderProgramObject);
				System.out.println("HAD: Shader Program Link Log = "+szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}

		//Get MVP uniform location
		modelMatrixUniform = GLES32.glGetUniformLocation(shaderProgramObject,"u_model_matrix");
		viewMatrixUniform = GLES32.glGetUniformLocation(shaderProgramObject,"u_view_matrix");
		projectionMatrixUniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_projection_matrix");
		doubleTapUniform = GLES32.glGetUniformLocation(shaderProgramObject,"u_double_tap");
		laUniform_White = GLES32.glGetUniformLocation(shaderProgramObject,"u_La_white");
		ldUniform_White = GLES32.glGetUniformLocation(shaderProgramObject,"u_Ld_white");
		lsUnifrom_White = GLES32.glGetUniformLocation(shaderProgramObject,"u_Ls_white");
		lightPositionUniform_White = GLES32.glGetUniformLocation(shaderProgramObject,"u_light_position_white");
		kaUniform = GLES32.glGetUniformLocation(shaderProgramObject,"u_Ka");
		kdUniform = GLES32.glGetUniformLocation(shaderProgramObject,"u_Kd");
		ksUniform = GLES32.glGetUniformLocation(shaderProgramObject,"u_Ks");
		materialShininessUniform = GLES32.glGetUniformLocation(shaderProgramObject,"u_material_shininess");

		//Vertices, Color,Shader Attribs, Vbo, Vao initializations
		sphere.getSphereVertexData(sphere_vertices,sphere_normals,sphere_textures,sphere_elements);
		iNumVertices = sphere.getNumberOfSphereVertices();
		iNumElements = sphere.getNumberOfSphereElements();

		/*****************Square*****************/
		GLES32.glGenVertexArrays(1,vao_sphere,0);
		GLES32.glBindVertexArray(vao_sphere[0]);

		/****************Sphere Position**************/
		GLES32.glGenBuffers(1,vbo_position,0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER,vbo_position[0]);

		ByteBuffer byteBuffer=ByteBuffer.allocateDirect(sphere_vertices.length*4);
		byteBuffer.order(ByteOrder.nativeOrder());
		FloatBuffer verticesBuffer=byteBuffer.asFloatBuffer();
		verticesBuffer.put(sphere_vertices);
		verticesBuffer.position(0);

		GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,sphere_vertices.length*4,verticesBuffer,GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(GLESMacros.HAD_ATTRIBUTE_VERTEX,3,GLES32.GL_FLOAT,false,0,0);

		GLES32.glEnableVertexAttribArray(GLESMacros.HAD_ATTRIBUTE_VERTEX);

		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER,0);

		/****************Sphere Color**************/
		GLES32.glGenBuffers(1,vbo_normal,0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER,vbo_normal[0]);

		byteBuffer=ByteBuffer.allocateDirect(sphere_normals.length*4);
		byteBuffer.order(ByteOrder.nativeOrder());
		FloatBuffer nomralBuffer=byteBuffer.asFloatBuffer();
		nomralBuffer.put(sphere_normals);
		nomralBuffer.position(0);

		GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,sphere_normals.length*4,nomralBuffer,GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(GLESMacros.HAD_ATTRIBUTE_NORMAL,3,GLES32.GL_FLOAT,false,0,0);

		GLES32.glEnableVertexAttribArray(GLESMacros.HAD_ATTRIBUTE_NORMAL);

		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER,0);

		/****************Sphere Elements************/
		GLES32.glGenBuffers(1,vbo_elements,0);
		GLES32.glBindBuffer(GLES32.GL_ELEMENT_ARRAY_BUFFER,vbo_elements[0]);

		byteBuffer=ByteBuffer.allocateDirect(sphere_elements.length*2);
		byteBuffer.order(ByteOrder.nativeOrder());
		ShortBuffer elementBuffer=byteBuffer.asShortBuffer();
		elementBuffer.put(sphere_elements);
		elementBuffer.position(0);

		GLES32.glBufferData(GLES32.GL_ELEMENT_ARRAY_BUFFER,sphere_elements.length*2,elementBuffer,GLES32.GL_STATIC_DRAW);

		GLES32.glBindBuffer(GLES32.GL_ELEMENT_ARRAY_BUFFER,0);

		GLES32.glBindVertexArray(0);

		//Enable DepthTest
		GLES32.glEnable(GLES32.GL_DEPTH_TEST);

		//Specify Depth test to be done
		GLES32.glDepthFunc(GLES32.GL_LEQUAL);

		//We will always cull the back faces for better performance
		//GLES32.glEnable(GLES32.GL_CULL_FACE);

		//Set the background frame color
		GLES32.glClearColor(0.25f,0.25f,0.25f,1.0f);

		doubleTap=0;

		//Set ProjectionMatrix to identity matrix
		Matrix.setIdentityM(perspectiveProjectionMatrix,0);
	}

	private void resize(int width,int height)
	{
		//Adjust the viewport based on geometry changes such as screen rotation
		GLES32.glViewport(0,0,width,height);

		//Perspective Projection
		Matrix.perspectiveM(perspectiveProjectionMatrix,0,45.0f,(float)width/(float)height,0.1f,100.0f);
	}

	public void draw()
	{
		//Draw background color
		GLES32.glClear(GLES32.GL_COLOR_BUFFER_BIT|GLES32.GL_DEPTH_BUFFER_BIT);

		//Use Shader Program
		GLES32.glUseProgram(shaderProgramObject);

		if(doubleTap == 1)
		{
			if(singleTap == 1)
			{
				light_position_White[1] = (float)Math.cos(angleSphere) * 100.0f;
				light_position_White[2] = (float)Math.sin(angleSphere) * 100.0f;
				light_position_White[0] = 0.0f;
			}
			else if(singleTap == 2)
			{
				light_position_White[0] = (float)Math.cos(angleSphere) * 100.0f;
				light_position_White[2] = (float)Math.sin(angleSphere) * 100.0f;
				light_position_White[1] = 0.0f;
			}
			else if(singleTap == 3)
			{
				light_position_White[0] = (float)Math.cos(angleSphere) * 100.0f;
				light_position_White[1] = (float)Math.sin(angleSphere) * 100.0f;
				light_position_White[2] = -2.0f;
			}
			if(singleTap == 0)
			{
				light_position_White[0] = 0.0f;
				light_position_White[1] = 0.0f;
				light_position_White[2] = 100.0f;
			}
			
			GLES32.glUniform1i(doubleTapUniform,1);

			GLES32.glUniform3fv(laUniform_White,1,light_ambient_White,0);
			GLES32.glUniform3fv(ldUniform_White,1,light_diffuse_White,0);
			GLES32.glUniform3fv(lsUnifrom_White,1,light_specular_White,0);
			GLES32.glUniform4fv(lightPositionUniform_White,1,light_position_White,0);

		}
		else
		{
			GLES32.glUniform1i(doubleTapUniform,0);
		}

		GLES32.glUniformMatrix4fv(projectionMatrixUniform,1,false,perspectiveProjectionMatrix,0);

		GLES32.glBindVertexArray(vao_sphere[0]);
		GLES32.glBindBuffer(GLES32.GL_ELEMENT_ARRAY_BUFFER,vbo_elements[0]);
		
		//set ModelView and ModelViewProjection matrices to identity
		
		Draw_Sphere_1();
		Draw_Sphere_2();
		Draw_Sphere_3();
		Draw_Sphere_4();
		Draw_Sphere_5();
		Draw_Sphere_6();
		Draw_Sphere_7();
		Draw_Sphere_8();
		Draw_Sphere_9();
		Draw_Sphere_10();
		Draw_Sphere_11();
		Draw_Sphere_12();
		Draw_Sphere_13();
		Draw_Sphere_14();
		Draw_Sphere_15();
		Draw_Sphere_16();
		Draw_Sphere_17();
		Draw_Sphere_18();
		Draw_Sphere_19();
		Draw_Sphere_20();
		Draw_Sphere_21();
		Draw_Sphere_22();
		Draw_Sphere_23();
		Draw_Sphere_24();

		GLES32.glBindBuffer(GLES32.GL_ELEMENT_ARRAY_BUFFER,0);

		GLES32.glBindVertexArray(0);

		//un-use shader program
		GLES32.glUseProgram(0);

		if(longPress == 1)
			update();
		//Like SwapBuffers() in Windows
		requestRender();
	}

	private void Draw_Sphere_1()
	{
		if(doubleTap == 1)
		{
			GLES32.glUniform3fv(kaUniform,1,material_ambient_1,0);
			GLES32.glUniform3fv(kdUniform,1,material_diffuse_1,0);
			GLES32.glUniform3fv(ksUniform,1,material_specular_1,0);
			GLES32.glUniform1f(materialShininessUniform,material_shininess_1);
		}
		Matrix.setIdentityM(modelMatrix,0);
		Matrix.setIdentityM(viewMatrix,0);
		Matrix.setIdentityM(scaleMatrix,0);

		Matrix.translateM(modelMatrix,0,-6.0f,-3.0f,-10.0f);

		Matrix.scaleM(scaleMatrix,0,1.5f,1.5f,1.5f);

		Matrix.multiplyMM(modelMatrix,0,modelMatrix,0,scaleMatrix,0);

		GLES32.glUniformMatrix4fv(modelMatrixUniform,1,false,modelMatrix,0);

		GLES32.glUniformMatrix4fv(viewMatrixUniform,1,false,viewMatrix,0);

		GLES32.glDrawElements(GLES32.GL_TRIANGLES,iNumElements,GLES32.GL_UNSIGNED_SHORT,0);
	}

	private void Draw_Sphere_2()
	{
		if(doubleTap == 1)
		{
			GLES32.glUniform3fv(kaUniform,1,material_ambient_2,0);
			GLES32.glUniform3fv(kdUniform,1,material_diffuse_2,0);
			GLES32.glUniform3fv(ksUniform,1,material_specular_2,0);
			GLES32.glUniform1f(materialShininessUniform,material_shininess_2);
		}
		Matrix.setIdentityM(modelMatrix,0);
		Matrix.setIdentityM(viewMatrix,0);
		Matrix.setIdentityM(scaleMatrix,0);

		Matrix.translateM(modelMatrix,0,-3.6f,-3.0f,-10.0f);

		Matrix.scaleM(scaleMatrix,0,1.5f,1.5f,1.5f);

		Matrix.multiplyMM(modelMatrix,0,modelMatrix,0,scaleMatrix,0);

		GLES32.glUniformMatrix4fv(modelMatrixUniform,1,false,modelMatrix,0);

		GLES32.glUniformMatrix4fv(viewMatrixUniform,1,false,viewMatrix,0);

		GLES32.glDrawElements(GLES32.GL_TRIANGLES,iNumElements,GLES32.GL_UNSIGNED_SHORT,0);
	}

	private void Draw_Sphere_3()
	{
		if(doubleTap == 1)
		{
			GLES32.glUniform3fv(kaUniform,1,material_ambient_3,0);
			GLES32.glUniform3fv(kdUniform,1,material_diffuse_3,0);
			GLES32.glUniform3fv(ksUniform,1,material_specular_3,0);
			GLES32.glUniform1f(materialShininessUniform,material_shininess_3);
		}
		Matrix.setIdentityM(modelMatrix,0);
		Matrix.setIdentityM(viewMatrix,0);
		Matrix.setIdentityM(scaleMatrix,0);

		Matrix.translateM(modelMatrix,0,-1.2f,-3.0f,-10.0f);

		Matrix.scaleM(scaleMatrix,0,1.5f,1.5f,1.5f);

		Matrix.multiplyMM(modelMatrix,0,modelMatrix,0,scaleMatrix,0);

		GLES32.glUniformMatrix4fv(modelMatrixUniform,1,false,modelMatrix,0);

		GLES32.glUniformMatrix4fv(viewMatrixUniform,1,false,viewMatrix,0);

		GLES32.glDrawElements(GLES32.GL_TRIANGLES,iNumElements,GLES32.GL_UNSIGNED_SHORT,0);
	}

	private void Draw_Sphere_4()
	{
		if(doubleTap == 1)
		{
			GLES32.glUniform3fv(kaUniform,1,material_ambient_4,0);
			GLES32.glUniform3fv(kdUniform,1,material_diffuse_4,0);
			GLES32.glUniform3fv(ksUniform,1,material_specular_4,0);
			GLES32.glUniform1f(materialShininessUniform,material_shininess_4);
		}
		Matrix.setIdentityM(modelMatrix,0);
		Matrix.setIdentityM(viewMatrix,0);
		Matrix.setIdentityM(scaleMatrix,0);

		Matrix.translateM(modelMatrix,0,1.2f,-3.0f,-10.0f);

		Matrix.scaleM(scaleMatrix,0,1.5f,1.5f,1.5f);

		Matrix.multiplyMM(modelMatrix,0,modelMatrix,0,scaleMatrix,0);

		GLES32.glUniformMatrix4fv(modelMatrixUniform,1,false,modelMatrix,0);

		GLES32.glUniformMatrix4fv(viewMatrixUniform,1,false,viewMatrix,0);

		GLES32.glDrawElements(GLES32.GL_TRIANGLES,iNumElements,GLES32.GL_UNSIGNED_SHORT,0);
	}

	private void Draw_Sphere_5()
	{
		if(doubleTap == 1)
		{
			GLES32.glUniform3fv(kaUniform,1,material_ambient_5,0);
			GLES32.glUniform3fv(kdUniform,1,material_diffuse_5,0);
			GLES32.glUniform3fv(ksUniform,1,material_specular_5,0);
			GLES32.glUniform1f(materialShininessUniform,material_shininess_5);
		}
		Matrix.setIdentityM(modelMatrix,0);
		Matrix.setIdentityM(viewMatrix,0);
		Matrix.setIdentityM(scaleMatrix,0);

		Matrix.translateM(modelMatrix,0,3.6f,-3.0f,-10.0f);

		Matrix.scaleM(scaleMatrix,0,1.5f,1.5f,1.5f);

		Matrix.multiplyMM(modelMatrix,0,modelMatrix,0,scaleMatrix,0);

		GLES32.glUniformMatrix4fv(modelMatrixUniform,1,false,modelMatrix,0);

		GLES32.glUniformMatrix4fv(viewMatrixUniform,1,false,viewMatrix,0);

		GLES32.glDrawElements(GLES32.GL_TRIANGLES,iNumElements,GLES32.GL_UNSIGNED_SHORT,0);
	}

	private void Draw_Sphere_6()
	{
		if(doubleTap == 1)
		{
			GLES32.glUniform3fv(kaUniform,1,material_ambient_6,0);
			GLES32.glUniform3fv(kdUniform,1,material_diffuse_6,0);
			GLES32.glUniform3fv(ksUniform,1,material_specular_6,0);
			GLES32.glUniform1f(materialShininessUniform,material_shininess_6);
		}
		Matrix.setIdentityM(modelMatrix,0);
		Matrix.setIdentityM(viewMatrix,0);
		Matrix.setIdentityM(scaleMatrix,0);

		Matrix.translateM(modelMatrix,0,6.0f,-3.0f,-10.0f);

		Matrix.scaleM(scaleMatrix,0,1.5f,1.5f,1.5f);

		Matrix.multiplyMM(modelMatrix,0,modelMatrix,0,scaleMatrix,0);

		GLES32.glUniformMatrix4fv(modelMatrixUniform,1,false,modelMatrix,0);

		GLES32.glUniformMatrix4fv(viewMatrixUniform,1,false,viewMatrix,0);

		GLES32.glDrawElements(GLES32.GL_TRIANGLES,iNumElements,GLES32.GL_UNSIGNED_SHORT,0);
	}

	private void Draw_Sphere_7()
	{
		if(doubleTap == 1)
		{
			GLES32.glUniform3fv(kaUniform,1,material_ambient_7,0);
			GLES32.glUniform3fv(kdUniform,1,material_diffuse_7,0);
			GLES32.glUniform3fv(ksUniform,1,material_specular_7,0);
			GLES32.glUniform1f(materialShininessUniform,material_shininess_7);
		}
		Matrix.setIdentityM(modelMatrix,0);
		Matrix.setIdentityM(viewMatrix,0);
		Matrix.setIdentityM(scaleMatrix,0);

		Matrix.translateM(modelMatrix,0,-6.0f,-1.0f,-10.0f);

		Matrix.scaleM(scaleMatrix,0,1.5f,1.5f,1.5f);

		Matrix.multiplyMM(modelMatrix,0,modelMatrix,0,scaleMatrix,0);

		GLES32.glUniformMatrix4fv(modelMatrixUniform,1,false,modelMatrix,0);

		GLES32.glUniformMatrix4fv(viewMatrixUniform,1,false,viewMatrix,0);

		GLES32.glDrawElements(GLES32.GL_TRIANGLES,iNumElements,GLES32.GL_UNSIGNED_SHORT,0);
	}

	private void Draw_Sphere_8()
	{
		if(doubleTap == 1)
		{
			GLES32.glUniform3fv(kaUniform,1,material_ambient_8,0);
			GLES32.glUniform3fv(kdUniform,1,material_diffuse_8,0);
			GLES32.glUniform3fv(ksUniform,1,material_specular_8,0);
			GLES32.glUniform1f(materialShininessUniform,material_shininess_8);
		}
		Matrix.setIdentityM(modelMatrix,0);
		Matrix.setIdentityM(viewMatrix,0);
		Matrix.setIdentityM(scaleMatrix,0);

		Matrix.translateM(modelMatrix,0,-3.6f,-1.0f,-10.0f);

		Matrix.scaleM(scaleMatrix,0,1.5f,1.5f,1.5f);

		Matrix.multiplyMM(modelMatrix,0,modelMatrix,0,scaleMatrix,0);

		GLES32.glUniformMatrix4fv(modelMatrixUniform,1,false,modelMatrix,0);

		GLES32.glUniformMatrix4fv(viewMatrixUniform,1,false,viewMatrix,0);

		GLES32.glDrawElements(GLES32.GL_TRIANGLES,iNumElements,GLES32.GL_UNSIGNED_SHORT,0);
	}

	private void Draw_Sphere_9()
	{
		if(doubleTap == 1)
		{
			GLES32.glUniform3fv(kaUniform,1,material_ambient_9,0);
			GLES32.glUniform3fv(kdUniform,1,material_diffuse_9,0);
			GLES32.glUniform3fv(ksUniform,1,material_specular_9,0);
			GLES32.glUniform1f(materialShininessUniform,material_shininess_9);
		}
		Matrix.setIdentityM(modelMatrix,0);
		Matrix.setIdentityM(viewMatrix,0);
		Matrix.setIdentityM(scaleMatrix,0);

		Matrix.translateM(modelMatrix,0,-1.2f,-1.0f,-10.0f);

		Matrix.scaleM(scaleMatrix,0,1.5f,1.5f,1.5f);

		Matrix.multiplyMM(modelMatrix,0,modelMatrix,0,scaleMatrix,0);

		GLES32.glUniformMatrix4fv(modelMatrixUniform,1,false,modelMatrix,0);

		GLES32.glUniformMatrix4fv(viewMatrixUniform,1,false,viewMatrix,0);

		GLES32.glDrawElements(GLES32.GL_TRIANGLES,iNumElements,GLES32.GL_UNSIGNED_SHORT,0);
	}

	private void Draw_Sphere_10()
	{
		if(doubleTap == 1)
		{
			GLES32.glUniform3fv(kaUniform,1,material_ambient_10,0);
			GLES32.glUniform3fv(kdUniform,1,material_diffuse_10,0);
			GLES32.glUniform3fv(ksUniform,1,material_specular_10,0);
			GLES32.glUniform1f(materialShininessUniform,material_shininess_10);
		}
		Matrix.setIdentityM(modelMatrix,0);
		Matrix.setIdentityM(viewMatrix,0);
		Matrix.setIdentityM(scaleMatrix,0);

		Matrix.translateM(modelMatrix,0,1.2f,-1.0f,-10.0f);

		Matrix.scaleM(scaleMatrix,0,1.5f,1.5f,1.5f);

		Matrix.multiplyMM(modelMatrix,0,modelMatrix,0,scaleMatrix,0);

		GLES32.glUniformMatrix4fv(modelMatrixUniform,1,false,modelMatrix,0);

		GLES32.glUniformMatrix4fv(viewMatrixUniform,1,false,viewMatrix,0);

		GLES32.glDrawElements(GLES32.GL_TRIANGLES,iNumElements,GLES32.GL_UNSIGNED_SHORT,0);
	}

	private void Draw_Sphere_11()
	{
		if(doubleTap == 1)
		{
			GLES32.glUniform3fv(kaUniform,1,material_ambient_11,0);
			GLES32.glUniform3fv(kdUniform,1,material_diffuse_11,0);
			GLES32.glUniform3fv(ksUniform,1,material_specular_11,0);
			GLES32.glUniform1f(materialShininessUniform,material_shininess_11);
		}
		Matrix.setIdentityM(modelMatrix,0);
		Matrix.setIdentityM(viewMatrix,0);
		Matrix.setIdentityM(scaleMatrix,0);

		Matrix.translateM(modelMatrix,0,3.6f,-1.0f,-10.0f);

		Matrix.scaleM(scaleMatrix,0,1.5f,1.5f,1.5f);

		Matrix.multiplyMM(modelMatrix,0,modelMatrix,0,scaleMatrix,0);

		GLES32.glUniformMatrix4fv(modelMatrixUniform,1,false,modelMatrix,0);

		GLES32.glUniformMatrix4fv(viewMatrixUniform,1,false,viewMatrix,0);

		GLES32.glDrawElements(GLES32.GL_TRIANGLES,iNumElements,GLES32.GL_UNSIGNED_SHORT,0);
	}

	private void Draw_Sphere_12()
	{
		if(doubleTap == 1)
		{
			GLES32.glUniform3fv(kaUniform,1,material_ambient_12,0);
			GLES32.glUniform3fv(kdUniform,1,material_diffuse_12,0);
			GLES32.glUniform3fv(ksUniform,1,material_specular_12,0);
			GLES32.glUniform1f(materialShininessUniform,material_shininess_12);
		}
		Matrix.setIdentityM(modelMatrix,0);
		Matrix.setIdentityM(viewMatrix,0);
		Matrix.setIdentityM(scaleMatrix,0);

		Matrix.translateM(modelMatrix,0,6.0f,-1.0f,-10.0f);

		Matrix.scaleM(scaleMatrix,0,1.5f,1.5f,1.5f);

		Matrix.multiplyMM(modelMatrix,0,modelMatrix,0,scaleMatrix,0);

		GLES32.glUniformMatrix4fv(modelMatrixUniform,1,false,modelMatrix,0);

		GLES32.glUniformMatrix4fv(viewMatrixUniform,1,false,viewMatrix,0);

		GLES32.glDrawElements(GLES32.GL_TRIANGLES,iNumElements,GLES32.GL_UNSIGNED_SHORT,0);
	}

	private void Draw_Sphere_13()
	{
		if(doubleTap == 1)
		{
			GLES32.glUniform3fv(kaUniform,1,material_ambient_13,0);
			GLES32.glUniform3fv(kdUniform,1,material_diffuse_13,0);
			GLES32.glUniform3fv(ksUniform,1,material_specular_13,0);
			GLES32.glUniform1f(materialShininessUniform,material_shininess_13);
		}
		Matrix.setIdentityM(modelMatrix,0);
		Matrix.setIdentityM(viewMatrix,0);
		Matrix.setIdentityM(scaleMatrix,0);

		Matrix.translateM(modelMatrix,0,-6.0f,1.0f,-10.0f);

		Matrix.scaleM(scaleMatrix,0,1.5f,1.5f,1.5f);

		Matrix.multiplyMM(modelMatrix,0,modelMatrix,0,scaleMatrix,0);

		GLES32.glUniformMatrix4fv(modelMatrixUniform,1,false,modelMatrix,0);

		GLES32.glUniformMatrix4fv(viewMatrixUniform,1,false,viewMatrix,0);

		GLES32.glDrawElements(GLES32.GL_TRIANGLES,iNumElements,GLES32.GL_UNSIGNED_SHORT,0);
	}

	private void Draw_Sphere_14()
	{
		if(doubleTap == 1)
		{
			GLES32.glUniform3fv(kaUniform,1,material_ambient_14,0);
			GLES32.glUniform3fv(kdUniform,1,material_diffuse_14,0);
			GLES32.glUniform3fv(ksUniform,1,material_specular_14,0);
			GLES32.glUniform1f(materialShininessUniform,material_shininess_14);
		}
		Matrix.setIdentityM(modelMatrix,0);
		Matrix.setIdentityM(viewMatrix,0);
		Matrix.setIdentityM(scaleMatrix,0);

		Matrix.translateM(modelMatrix,0,-3.6f,1.0f,-10.0f);

		Matrix.scaleM(scaleMatrix,0,1.5f,1.5f,1.5f);

		Matrix.multiplyMM(modelMatrix,0,modelMatrix,0,scaleMatrix,0);

		GLES32.glUniformMatrix4fv(modelMatrixUniform,1,false,modelMatrix,0);

		GLES32.glUniformMatrix4fv(viewMatrixUniform,1,false,viewMatrix,0);

		GLES32.glDrawElements(GLES32.GL_TRIANGLES,iNumElements,GLES32.GL_UNSIGNED_SHORT,0);
	}

	private void Draw_Sphere_15()
	{
		if(doubleTap == 1)
		{
			GLES32.glUniform3fv(kaUniform,1,material_ambient_15,0);
			GLES32.glUniform3fv(kdUniform,1,material_diffuse_15,0);
			GLES32.glUniform3fv(ksUniform,1,material_specular_15,0);
			GLES32.glUniform1f(materialShininessUniform,material_shininess_15);
		}
		Matrix.setIdentityM(modelMatrix,0);
		Matrix.setIdentityM(viewMatrix,0);
		Matrix.setIdentityM(scaleMatrix,0);

		Matrix.translateM(modelMatrix,0,-1.2f,1.0f,-10.0f);

		Matrix.scaleM(scaleMatrix,0,1.5f,1.5f,1.5f);

		Matrix.multiplyMM(modelMatrix,0,modelMatrix,0,scaleMatrix,0);

		GLES32.glUniformMatrix4fv(modelMatrixUniform,1,false,modelMatrix,0);

		GLES32.glUniformMatrix4fv(viewMatrixUniform,1,false,viewMatrix,0);

		GLES32.glDrawElements(GLES32.GL_TRIANGLES,iNumElements,GLES32.GL_UNSIGNED_SHORT,0);
	}

	private void Draw_Sphere_16()
	{
		if(doubleTap == 1)
		{
			GLES32.glUniform3fv(kaUniform,1,material_ambient_16,0);
			GLES32.glUniform3fv(kdUniform,1,material_diffuse_16,0);
			GLES32.glUniform3fv(ksUniform,1,material_specular_16,0);
			GLES32.glUniform1f(materialShininessUniform,material_shininess_16);
		}
		Matrix.setIdentityM(modelMatrix,0);
		Matrix.setIdentityM(viewMatrix,0);
		Matrix.setIdentityM(scaleMatrix,0);

		Matrix.translateM(modelMatrix,0,1.2f,1.0f,-10.0f);

		Matrix.scaleM(scaleMatrix,0,1.5f,1.5f,1.5f);

		Matrix.multiplyMM(modelMatrix,0,modelMatrix,0,scaleMatrix,0);

		GLES32.glUniformMatrix4fv(modelMatrixUniform,1,false,modelMatrix,0);

		GLES32.glUniformMatrix4fv(viewMatrixUniform,1,false,viewMatrix,0);

		GLES32.glDrawElements(GLES32.GL_TRIANGLES,iNumElements,GLES32.GL_UNSIGNED_SHORT,0);
	}

	private void Draw_Sphere_17()
	{
		if(doubleTap == 1)
		{
			GLES32.glUniform3fv(kaUniform,1,material_ambient_17,0);
			GLES32.glUniform3fv(kdUniform,1,material_diffuse_17,0);
			GLES32.glUniform3fv(ksUniform,1,material_specular_17,0);
			GLES32.glUniform1f(materialShininessUniform,material_shininess_17);
		}
		Matrix.setIdentityM(modelMatrix,0);
		Matrix.setIdentityM(viewMatrix,0);
		Matrix.setIdentityM(scaleMatrix,0);

		Matrix.translateM(modelMatrix,0,3.6f,1.0f,-10.0f);

		Matrix.scaleM(scaleMatrix,0,1.5f,1.5f,1.5f);

		Matrix.multiplyMM(modelMatrix,0,modelMatrix,0,scaleMatrix,0);

		GLES32.glUniformMatrix4fv(modelMatrixUniform,1,false,modelMatrix,0);

		GLES32.glUniformMatrix4fv(viewMatrixUniform,1,false,viewMatrix,0);

		GLES32.glDrawElements(GLES32.GL_TRIANGLES,iNumElements,GLES32.GL_UNSIGNED_SHORT,0);
	}

	private void Draw_Sphere_18()
	{
		if(doubleTap == 1)
		{
			GLES32.glUniform3fv(kaUniform,1,material_ambient_18,0);
			GLES32.glUniform3fv(kdUniform,1,material_diffuse_18,0);
			GLES32.glUniform3fv(ksUniform,1,material_specular_18,0);
			GLES32.glUniform1f(materialShininessUniform,material_shininess_18);
		}
		Matrix.setIdentityM(modelMatrix,0);
		Matrix.setIdentityM(viewMatrix,0);
		Matrix.setIdentityM(scaleMatrix,0);

		Matrix.translateM(modelMatrix,0,6.0f,1.0f,-10.0f);

		Matrix.scaleM(scaleMatrix,0,1.5f,1.5f,1.5f);

		Matrix.multiplyMM(modelMatrix,0,modelMatrix,0,scaleMatrix,0);

		GLES32.glUniformMatrix4fv(modelMatrixUniform,1,false,modelMatrix,0);

		GLES32.glUniformMatrix4fv(viewMatrixUniform,1,false,viewMatrix,0);

		GLES32.glDrawElements(GLES32.GL_TRIANGLES,iNumElements,GLES32.GL_UNSIGNED_SHORT,0);
	}

	private void Draw_Sphere_19()
	{
		if(doubleTap == 1)
		{
			GLES32.glUniform3fv(kaUniform,1,material_ambient_19,0);
			GLES32.glUniform3fv(kdUniform,1,material_diffuse_19,0);
			GLES32.glUniform3fv(ksUniform,1,material_specular_19,0);
			GLES32.glUniform1f(materialShininessUniform,material_shininess_19);
		}
		Matrix.setIdentityM(modelMatrix,0);
		Matrix.setIdentityM(viewMatrix,0);
		Matrix.setIdentityM(scaleMatrix,0);

		Matrix.translateM(modelMatrix,0,-6.0f,3.0f,-10.0f);

		Matrix.scaleM(scaleMatrix,0,1.5f,1.5f,1.5f);

		Matrix.multiplyMM(modelMatrix,0,modelMatrix,0,scaleMatrix,0);

		GLES32.glUniformMatrix4fv(modelMatrixUniform,1,false,modelMatrix,0);

		GLES32.glUniformMatrix4fv(viewMatrixUniform,1,false,viewMatrix,0);

		GLES32.glDrawElements(GLES32.GL_TRIANGLES,iNumElements,GLES32.GL_UNSIGNED_SHORT,0);
	}

	private void Draw_Sphere_20()
	{
		if(doubleTap == 1)
		{
			GLES32.glUniform3fv(kaUniform,1,material_ambient_20,0);
			GLES32.glUniform3fv(kdUniform,1,material_diffuse_20,0);
			GLES32.glUniform3fv(ksUniform,1,material_specular_20,0);
			GLES32.glUniform1f(materialShininessUniform,material_shininess_20);
		}
		Matrix.setIdentityM(modelMatrix,0);
		Matrix.setIdentityM(viewMatrix,0);
		Matrix.setIdentityM(scaleMatrix,0);

		Matrix.translateM(modelMatrix,0,-3.6f,3.0f,-10.0f);

		Matrix.scaleM(scaleMatrix,0,1.5f,1.5f,1.5f);

		Matrix.multiplyMM(modelMatrix,0,modelMatrix,0,scaleMatrix,0);

		GLES32.glUniformMatrix4fv(modelMatrixUniform,1,false,modelMatrix,0);

		GLES32.glUniformMatrix4fv(viewMatrixUniform,1,false,viewMatrix,0);

		GLES32.glDrawElements(GLES32.GL_TRIANGLES,iNumElements,GLES32.GL_UNSIGNED_SHORT,0);
	}

	private void Draw_Sphere_21()
	{
		if(doubleTap == 1)
		{
			GLES32.glUniform3fv(kaUniform,1,material_ambient_21,0);
			GLES32.glUniform3fv(kdUniform,1,material_diffuse_21,0);
			GLES32.glUniform3fv(ksUniform,1,material_specular_21,0);
			GLES32.glUniform1f(materialShininessUniform,material_shininess_21);
		}
		Matrix.setIdentityM(modelMatrix,0);
		Matrix.setIdentityM(viewMatrix,0);
		Matrix.setIdentityM(scaleMatrix,0);

		Matrix.translateM(modelMatrix,0,-1.2f,3.0f,-10.0f);

		Matrix.scaleM(scaleMatrix,0,1.5f,1.5f,1.5f);

		Matrix.multiplyMM(modelMatrix,0,modelMatrix,0,scaleMatrix,0);

		GLES32.glUniformMatrix4fv(modelMatrixUniform,1,false,modelMatrix,0);

		GLES32.glUniformMatrix4fv(viewMatrixUniform,1,false,viewMatrix,0);

		GLES32.glDrawElements(GLES32.GL_TRIANGLES,iNumElements,GLES32.GL_UNSIGNED_SHORT,0);
	}

	private void Draw_Sphere_22()
	{
		if(doubleTap == 1)
		{
			GLES32.glUniform3fv(kaUniform,1,material_ambient_22,0);
			GLES32.glUniform3fv(kdUniform,1,material_diffuse_22,0);
			GLES32.glUniform3fv(ksUniform,1,material_specular_22,0);
			GLES32.glUniform1f(materialShininessUniform,material_shininess_22);
		}
		Matrix.setIdentityM(modelMatrix,0);
		Matrix.setIdentityM(viewMatrix,0);
		Matrix.setIdentityM(scaleMatrix,0);

		Matrix.translateM(modelMatrix,0,1.2f,3.0f,-10.0f);

		Matrix.scaleM(scaleMatrix,0,1.5f,1.5f,1.5f);

		Matrix.multiplyMM(modelMatrix,0,modelMatrix,0,scaleMatrix,0);

		GLES32.glUniformMatrix4fv(modelMatrixUniform,1,false,modelMatrix,0);

		GLES32.glUniformMatrix4fv(viewMatrixUniform,1,false,viewMatrix,0);

		GLES32.glDrawElements(GLES32.GL_TRIANGLES,iNumElements,GLES32.GL_UNSIGNED_SHORT,0);
	}

	private void Draw_Sphere_23()
	{
		if(doubleTap == 1)
		{
			GLES32.glUniform3fv(kaUniform,1,material_ambient_23,0);
			GLES32.glUniform3fv(kdUniform,1,material_diffuse_23,0);
			GLES32.glUniform3fv(ksUniform,1,material_specular_23,0);
			GLES32.glUniform1f(materialShininessUniform,material_shininess_23);
		}
		Matrix.setIdentityM(modelMatrix,0);
		Matrix.setIdentityM(viewMatrix,0);
		Matrix.setIdentityM(scaleMatrix,0);

		Matrix.translateM(modelMatrix,0,3.6f,3.0f,-10.0f);

		Matrix.scaleM(scaleMatrix,0,1.5f,1.5f,1.5f);

		Matrix.multiplyMM(modelMatrix,0,modelMatrix,0,scaleMatrix,0);

		GLES32.glUniformMatrix4fv(modelMatrixUniform,1,false,modelMatrix,0);

		GLES32.glUniformMatrix4fv(viewMatrixUniform,1,false,viewMatrix,0);

		GLES32.glDrawElements(GLES32.GL_TRIANGLES,iNumElements,GLES32.GL_UNSIGNED_SHORT,0);
	}

	private void Draw_Sphere_24()
	{
		if(doubleTap == 1)
		{
			GLES32.glUniform3fv(kaUniform,1,material_ambient_24,0);
			GLES32.glUniform3fv(kdUniform,1,material_diffuse_24,0);
			GLES32.glUniform3fv(ksUniform,1,material_specular_24,0);
			GLES32.glUniform1f(materialShininessUniform,material_shininess_24);
		}
		Matrix.setIdentityM(modelMatrix,0);
		Matrix.setIdentityM(viewMatrix,0);
		Matrix.setIdentityM(scaleMatrix,0);

		Matrix.translateM(modelMatrix,0,6.0f,3.0f,-10.0f);

		Matrix.scaleM(scaleMatrix,0,1.5f,1.5f,1.5f);

		Matrix.multiplyMM(modelMatrix,0,modelMatrix,0,scaleMatrix,0);

		GLES32.glUniformMatrix4fv(modelMatrixUniform,1,false,modelMatrix,0);

		GLES32.glUniformMatrix4fv(viewMatrixUniform,1,false,viewMatrix,0);

		GLES32.glDrawElements(GLES32.GL_TRIANGLES,iNumElements,GLES32.GL_UNSIGNED_SHORT,0);
	}
	

	private void update()
	{
			angleSphere=angleSphere+0.05f;
			if(angleSphere>=360.0f)
				angleSphere=angleSphere-360.0f;
	}

	public void uninitialize()
	{
		if(vao_sphere[0] != 0)
		{
			GLES32.glDeleteVertexArrays(1,vao_sphere,0);
			vao_sphere[0]=0;
		}

		if(vbo_position[0] != 0)
		{
			GLES32.glDeleteBuffers(1,vbo_position,0);
			vbo_position[0]=0;
		}

		if(vbo_normal[0] != 0)
		{
			GLES32.glDeleteBuffers(1,vbo_normal,0);
			vbo_normal[0]=0;
		}

		if(vbo_elements[0] != 0)
		{
			GLES32.glDeleteBuffers(1,vbo_elements,0);
			vbo_elements[0]=0;
		}

		if(shaderProgramObject != 0)
		{
			if(vertexShaderObject != 0)
			{
				GLES32.glDetachShader(shaderProgramObject,vertexShaderObject);
				GLES32.glDeleteShader(vertexShaderObject);
				vertexShaderObject=0;
			}

			if(fragmentShaderObject != 0)
			{
				GLES32.glDetachShader(shaderProgramObject,fragmentShaderObject);
				GLES32.glDeleteShader(fragmentShaderObject);
				fragmentShaderObject=0;
			}
		}

		if(shaderProgramObject != 0)
		{
			GLES32.glDeleteProgram(shaderProgramObject);
			shaderProgramObject=0;
		}
	}
}