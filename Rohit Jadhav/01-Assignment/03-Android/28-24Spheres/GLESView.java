package com.rohit_r_jadhav.light_on_24sphere;

import android.opengl.GLSurfaceView;
import android.opengl.GLES32;
import android.opengl.Matrix;

import javax.microedition.khronos.opengles.GL10;
import javax.microedition.khronos.egl.EGLConfig;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;

import android.view.MotionEvent;
import android.view.GestureDetector;
import android.view.GestureDetector.OnDoubleTapListener;
import android.view.GestureDetector.OnGestureListener;
import android.content.Context;

public class GLESView extends GLSurfaceView implements GLSurfaceView.Renderer, OnDoubleTapListener, OnGestureListener{


	final int PER_VERTEX = 1;
	final int PER_FRAGMENT = 2;
	final int X_ROT = 3;
	final int Y_ROT = 4;
	final int Z_ROT = 5;


	GestureDetector gestureDetector;
	Context context;
	boolean bLights = false;
	int iWhichLight = PER_VERTEX;
	int iWhichRotation = X_ROT;

	GLESView(Context drawingContext){
		super(drawingContext);
		context = drawingContext;

		gestureDetector = new GestureDetector(drawingContext, this, null, false);
		gestureDetector.setOnDoubleTapListener(this);

		setEGLContextClientVersion(3);
		setRenderer(this);
		setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
	}

	@Override
	public boolean onTouchEvent(MotionEvent e){
		int action = e.getAction();
		if(!gestureDetector.onTouchEvent(e))
			super.onTouchEvent(e);
		return(true);
	}

	

	/********** Methods from OnDoubleTapListener **********/
	@Override
	public boolean onDoubleTap(MotionEvent e){
		if(iWhichLight == PER_VERTEX)
			iWhichLight = PER_FRAGMENT;
		else
			iWhichLight = PER_VERTEX;
		return(true);
	}

	@Override
	public boolean onDoubleTapEvent(MotionEvent e){
		return(true);
	}

	@Override
	public boolean onSingleTapConfirmed(MotionEvent e){
		if(bLights == false)
			bLights = true; 
		else{

			if(iWhichRotation == X_ROT){
				lightPosition[0] = 0.0f;
				lightPosition[1] = 0.0f;
				lightPosition[2] = 0.0f;
				iWhichRotation = Y_ROT;
			}
			else if(iWhichRotation == Y_ROT){
				lightPosition[0] = 0.0f;
				lightPosition[1] = 0.0f;
				lightPosition[2] = 0.0f;
				iWhichRotation = Z_ROT;
			}
			else if(iWhichRotation == Z_ROT){
				lightPosition[0] = 0.0f;
				lightPosition[1] = 0.0f;
				lightPosition[2] = 0.0f;
				iWhichRotation = X_ROT;
				bLights = false;
			}
		}
		
			return(true);
	}



	/********** Methods from OnGestureListener **********/
	@Override
	public boolean onDown(MotionEvent e){
		return(true);
	}

	@Override
	public boolean onSingleTapUp(MotionEvent e){
		return(true);
	}

	@Override
	public boolean onFling(MotionEvent e1, MotionEvent e2, float velocityX, float velocityY){
		return(true);
	}

	@Override
	public void onLongPress(MotionEvent e){

	}

	@Override
	public void onShowPress(MotionEvent e){

	}

	@Override
	public boolean onScroll(MotionEvent e1, MotionEvent e2, float distanceX, float distanceY){
		uninitialize();
		System.exit(0);
		return(true);
	}


	/********** Methods from GLSurfaceView.Renderer **********/
	@Override
	public void onSurfaceCreated(GL10 gl, EGLConfig config){

		String version = gl.glGetString(GL10.GL_VERSION);
		String glsl_version = gl.glGetString(GLES32.GL_SHADING_LANGUAGE_VERSION);
		String renderer = gl.glGetString(GL10.GL_RENDERER);
		String vendor = gl.glGetString(GL10.GL_VENDOR);

		System.out.println("RTR: OpenGL Version: " + version);
		System.out.println("RTR: OpenGLSL Verion: " + glsl_version);
		System.out.println("RTR: Renderer: " + renderer);
		System.out.println("RTR: Vendor: "+ vendor);

		initialize();
	}

	@Override
	public void onSurfaceChanged(GL10 unused, int width, int height){
		resize(width, height);
	}

	@Override
	public void onDrawFrame(GL10 unused){
		update();
		display();
	}



	//For Shader
	private int vertexShaderObject_PV;
	private int fragmentShaderObject_PV;
	private int shaderProgramObject_PV;

	private int vertexShaderObject_PF;
	private int fragmentShaderObject_PF;
	private int shaderProgramObject_PF;

	//For Sphere
	private int vao_Sphere[] = new int[1];
	private int vbo_Sphere_Position[] = new int[1];
	private int vbo_Sphere_Normal[] = new int[1];
	private int vbo_Sphere_Element[] = new int[1];
	int numVertices;
	int numElements;


	//For Viewport
	private int iViewPortNo = 1;
	private int viewPortWidth = 0;
	private int viewPortHeight = 0;



	//For Projection
	float perspectiveProjectionMatrix[] = new float[4*4];

	//For Uniform
	//Per Vertex
	private int modelMatrixUniform_PV;
	private int viewMatrixUniform_PV;
	private int projectionMatrixUniform_PV;
	private int la_Uniform_PV;
	private int ld_Uniform_PV;
	private int ls_Uniform_PV;
	private int lightPositionUniform_PV;
	private int ka_Uniform_PV;
	private int kd_Uniform_PV;
	private int ks_Uniform_PV;
	private int shininessUniform_PV;
	private int LKeyPressUniform_PV;

	//Per Fragment
	private int modelMatrixUniform_PF;
	private int viewMatrixUniform_PF;
	private int projectionMatrixUniform_PF;
	private int la_Uniform_PF;
	private int ld_Uniform_PF;
	private int ls_Uniform_PF;
	private int lightPositionUniform_PF;
	private int ka_Uniform_PF;
	private int kd_Uniform_PF;
	private int ks_Uniform_PF;
	private int shininessUniform_PF;
	private int LKeyPressUniform_PF;

	//For Lights
	float lightAmbient[] = new float[]{0.0f, 0.0f, 0.0f, 0.0f};
	float lightDiffuse[] = new float[] {1.0f, 1.0f, 1.0f, 1.0f};
	float lightSpecular[] = new float[]{1.0f, 1.0f, 1.0f, 1.0f};
	float lightPosition[] = new float[]{0.0f, 0.0f, 0.0f, 1.0f};
	float angle_X = 0.0f;
	float angle_Y = 0.0f;
	float angle_Z = 0.0f;



	private void initialize(){


		/********************** PER VERTEX LIGHTING ********************/

		vertexShaderObject_PV = GLES32.glCreateShader(GLES32.GL_VERTEX_SHADER);

		final String vertexShaderCode_PV = String.format(
				"#version 320 es" +
				"\n" +
				"in vec4 vPosition;" +
				"in vec3 vNormal;" +
				"uniform mat4 u_model_matrix;" +
				"uniform mat4 u_view_matrix;" +
				"uniform mat4 u_projection_matrix;" +
				"uniform vec3 u_la;" +
				"uniform vec3 u_ld;" +
				"uniform vec3 u_ls;" +
				"uniform vec4 u_light_position;" +
				"uniform vec3 u_ka;" +
				"uniform vec3 u_kd;" +
				"uniform vec3 u_ks;" +
				"uniform float u_shininess;" +
				"uniform int u_LKeyPress;" +
				"out vec3 phongLight;" +
				"void main(void)" +
				"{" +
					"if(u_LKeyPress == 1)" +
					"{" +
						"vec4 eyeCoordinate = u_view_matrix * u_model_matrix * vPosition;" +
						
						"vec3 lightDirection = normalize(vec3(u_light_position - eyeCoordinate));" +
						"mat3 normalMatrix = mat3(u_view_matrix * u_model_matrix);" +
						"vec3 tNormal = normalize(vec3(normalMatrix * vNormal));" +
						"float S_Dot_N = max(dot(lightDirection, tNormal), 0.0);" +
						
						"vec3 viewer = normalize(vec3(-eyeCoordinate.xyz));" +
						"vec3 reflection = reflect(-lightDirection, tNormal);" +
						"float R_Dot_V = max(dot(reflection, viewer), 0.0);" +

						"vec3 ambient = u_la * u_ka;" +
						"vec3 diffuse = u_ld * u_kd * S_Dot_N;"+
						"vec3 specular = u_ls * u_ks * pow(R_Dot_V, u_shininess);" +
						"phongLight = ambient + diffuse + specular;" +	
					"}" +
					"else" +
					"{" +
						"phongLight = vec3(1.0, 1.0, 1.0);" +
					"}"+

					"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" +

				"}"  
			);


		GLES32.glShaderSource(vertexShaderObject_PV, vertexShaderCode_PV);

		GLES32.glCompileShader(vertexShaderObject_PV);

		int iShaderCompileStatus[] = new int[1];
		int iInfoLogLength[] = new int[1];
		String szInfoLog = null;

		GLES32.glGetShaderiv(vertexShaderObject_PV, GLES32.GL_COMPILE_STATUS, iShaderCompileStatus, 0);
		if(iShaderCompileStatus[0] == GLES32.GL_FALSE){
			GLES32.glGetShaderiv(vertexShaderObject_PV, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if(iInfoLogLength[0] > 0){
				szInfoLog = GLES32.glGetShaderInfoLog(vertexShaderObject_PV);
				System.out.println("RTR: Per Vertex Lighting Vertex Shader Compilation Error: "+ szInfoLog);
				szInfoLog = null;
				uninitialize();
				System.exit(1);
			}
		}




		fragmentShaderObject_PV = GLES32.glCreateShader(GLES32.GL_FRAGMENT_SHADER);
		
		final String fragmentShaderCode_PV = String.format(
				"#version 320 es" +
				"\n" +
				"precision highp float;" +
				"in vec3 phongLight;" +
				"out vec4 FragColor;" +
				"void main(void)" +
				"{" +
					"FragColor = vec4(phongLight, 0.0f);" +
				"}"
			);

		GLES32.glShaderSource(fragmentShaderObject_PV, fragmentShaderCode_PV);
		GLES32.glCompileShader(fragmentShaderObject_PV);

		iShaderCompileStatus[0] = 0;
		iInfoLogLength[0] = 0;
		szInfoLog = null;

		GLES32.glGetShaderiv(fragmentShaderObject_PV, GLES32.GL_COMPILE_STATUS, iShaderCompileStatus, 0);
		if(iShaderCompileStatus[0] == GLES32.GL_FALSE){
			GLES32.glGetShaderiv(fragmentShaderObject_PV, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if(iInfoLogLength[0] > 0){
				szInfoLog = GLES32.glGetShaderInfoLog(fragmentShaderObject_PV);
				System.out.println("RTR: Per Vertex Lighting Fragment Shader Compilation Error: "+ szInfoLog);
				szInfoLog = null;
				uninitialize();
				System.exit(1);
			}
		}



		shaderProgramObject_PV = GLES32.glCreateProgram();

		GLES32.glAttachShader(shaderProgramObject_PV, vertexShaderObject_PV);
		GLES32.glAttachShader(shaderProgramObject_PV, fragmentShaderObject_PV);

		GLES32.glBindAttribLocation(shaderProgramObject_PV, GLESMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
		GLES32.glBindAttribLocation(shaderProgramObject_PV, GLESMacros.AMC_ATTRIBUTE_NORMAL, "vNormal");

		GLES32.glLinkProgram(shaderProgramObject_PV);

		int iProgramLinkStatus[] = new int[1];
		iInfoLogLength[0] = 0;
		szInfoLog = null;

		GLES32.glGetProgramiv(shaderProgramObject_PV, GLES32.GL_LINK_STATUS, iProgramLinkStatus, 0);
		if(iProgramLinkStatus[0] == GLES32.GL_FALSE){
			GLES32.glGetProgramiv(shaderProgramObject_PV, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if(iInfoLogLength[0] > 0){
				szInfoLog = GLES32.glGetProgramInfoLog(shaderProgramObject_PV);
				System.out.println("RTR: Per Vertex Lighting Shader Program Linking Error: " + szInfoLog);
				szInfoLog = null;
				uninitialize();
				System.exit(1);
			} 
		}



		modelMatrixUniform_PV = GLES32.glGetUniformLocation(shaderProgramObject_PV, "u_model_matrix");
		viewMatrixUniform_PV = GLES32.glGetUniformLocation(shaderProgramObject_PV, "u_view_matrix");
		projectionMatrixUniform_PV = GLES32.glGetUniformLocation(shaderProgramObject_PV, "u_projection_matrix");
		la_Uniform_PV = GLES32.glGetUniformLocation(shaderProgramObject_PV, "u_la");
		ld_Uniform_PV = GLES32.glGetUniformLocation(shaderProgramObject_PV, "u_ld");
		ls_Uniform_PV = GLES32.glGetUniformLocation(shaderProgramObject_PV, "u_ls");
		lightPositionUniform_PV = GLES32.glGetUniformLocation(shaderProgramObject_PV, "u_light_position");

		ka_Uniform_PV = GLES32.glGetUniformLocation(shaderProgramObject_PV, "u_ka");
		kd_Uniform_PV = GLES32.glGetUniformLocation(shaderProgramObject_PV, "u_kd");
		ks_Uniform_PV= GLES32.glGetUniformLocation(shaderProgramObject_PV, "u_ks");
		shininessUniform_PV = GLES32.glGetUniformLocation(shaderProgramObject_PV, "u_shininess");
		LKeyPressUniform_PV = GLES32.glGetUniformLocation(shaderProgramObject_PV, "u_LKeyPress");


	





		/********************** PER FRAGMENT LIGHTING ********************/


		vertexShaderObject_PF = GLES32.glCreateShader(GLES32.GL_VERTEX_SHADER);

		final String vertexShaderCode_PF = String.format(
				"#version 320 es" +
				"\n" +
				"in vec4 vPosition;" +
				"in vec3 vNormal;" +
				"uniform mat4 u_model_matrix;" +
				"uniform mat4 u_view_matrix;" +
				"uniform mat4 u_projection_matrix;" +
				"uniform vec4 u_light_position;" +
				"out vec3 lightDirection_VS;" +
				"out vec3 tNormal_VS;" +
				"out vec3 viewer_VS;" +
				"void main(void)" +
				"{" +
					"vec4 eyeCoordinate = u_view_matrix * u_model_matrix * vPosition;" +
					"lightDirection_VS = vec3(u_light_position - eyeCoordinate);" +
					"mat3 normalMatrix = mat3(u_view_matrix * u_model_matrix);" +
					"tNormal_VS = vec3(normalMatrix * vNormal);" +
					"viewer_VS = vec3(-eyeCoordinate.xyz);" +
					"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" +
				"}"
			);


		GLES32.glShaderSource(vertexShaderObject_PF, vertexShaderCode_PF);

		GLES32.glCompileShader(vertexShaderObject_PF);

		iShaderCompileStatus[0] = 0;
		iInfoLogLength[0] = 0;
		szInfoLog = null;

		GLES32.glGetShaderiv(vertexShaderObject_PF, GLES32.GL_COMPILE_STATUS, iShaderCompileStatus, 0);
		if(iShaderCompileStatus[0] == GLES32.GL_FALSE){
			GLES32.glGetShaderiv(vertexShaderObject_PF, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if(iInfoLogLength[0] > 0){
				szInfoLog = GLES32.glGetShaderInfoLog(vertexShaderObject_PF);
				System.out.println("RTR: Per Fragment Lighting Vertex Shader Compilation Error: "+ szInfoLog);
				szInfoLog = null;
				uninitialize();
				System.exit(1);
			}
		}




		fragmentShaderObject_PF = GLES32.glCreateShader(GLES32.GL_FRAGMENT_SHADER);
		
		final String fragmentShaderCode_PF = String.format(
				"#version 320 es" +
				"\n" +
				"precision highp float;" +
				"in vec3 lightDirection_VS;" +
				"in vec3 tNormal_VS;" +
				"in vec3 viewer_VS;" +
				"uniform vec3 u_la;" +
				"uniform vec3 u_ld;" +
				"uniform vec3 u_ls;" +
				"uniform vec3 u_ka;" +
				"uniform vec3 u_kd;" +
				"uniform vec3 u_ks;" +
				"uniform float u_shininess;" +
				"uniform int u_LKeyPress;" +
				"out vec4 FragColor;" +
				"void main(void)" +
				"{" +
					"vec3 phongLight;" +

					"if(u_LKeyPress == 1)" +
					"{" +
						"vec3 normalizeLightDirection = normalize(lightDirection_VS);" +
						"vec3 normalizeTNormal = normalize(tNormal_VS);" +
						"float S_Dot_N = max(dot(normalizeLightDirection, normalizeTNormal), 0.0);" +

						"vec3 normalizeViewer = normalize(viewer_VS);" +
						"vec3 Reflection = reflect(-normalizeLightDirection, normalizeTNormal);" +
						"float R_Dot_V = max(dot(Reflection, normalizeViewer), 0.0);" +

						"vec3 ambient = u_la * u_ka;" +
						"vec3 diffuse = u_ld * u_kd * S_Dot_N;"  +
						"vec3 specular = u_ls * u_ks * pow(R_Dot_V, u_shininess);" +
						"phongLight = ambient + diffuse + specular;" +
					"}" +
					"else" +
					"{" +
						"phongLight = vec3(1.0, 1.0, 1.0);" +
					"}" +

					"FragColor = vec4(phongLight, 0.0);" +
				"}"	
			);

		GLES32.glShaderSource(fragmentShaderObject_PF, fragmentShaderCode_PF);
		GLES32.glCompileShader(fragmentShaderObject_PF);

		iShaderCompileStatus[0] = 0;
		iInfoLogLength[0] = 0;
		szInfoLog = null;

		GLES32.glGetShaderiv(fragmentShaderObject_PF, GLES32.GL_COMPILE_STATUS, iShaderCompileStatus, 0);
		if(iShaderCompileStatus[0] == GLES32.GL_FALSE){
			GLES32.glGetShaderiv(fragmentShaderObject_PF, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if(iInfoLogLength[0] > 0){
				szInfoLog = GLES32.glGetShaderInfoLog(fragmentShaderObject_PF);
				System.out.println("RTR: Per Fragment Lighting Fragment Shader Compilation Error: "+ szInfoLog);
				szInfoLog = null;
				uninitialize();
				System.exit(1);
			}
		}



		shaderProgramObject_PF = GLES32.glCreateProgram();

		GLES32.glAttachShader(shaderProgramObject_PF, vertexShaderObject_PF);
		GLES32.glAttachShader(shaderProgramObject_PF, fragmentShaderObject_PF);

		GLES32.glBindAttribLocation(shaderProgramObject_PF, GLESMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
		GLES32.glBindAttribLocation(shaderProgramObject_PF, GLESMacros.AMC_ATTRIBUTE_NORMAL, "vNormal");

		GLES32.glLinkProgram(shaderProgramObject_PF);

		iProgramLinkStatus[0] = 0;
		iInfoLogLength[0] = 0;
		szInfoLog = null;

		GLES32.glGetProgramiv(shaderProgramObject_PF, GLES32.GL_LINK_STATUS, iProgramLinkStatus, 0);
		if(iProgramLinkStatus[0] == GLES32.GL_FALSE){
			GLES32.glGetProgramiv(shaderProgramObject_PF, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if(iInfoLogLength[0] > 0){
				szInfoLog = GLES32.glGetProgramInfoLog(shaderProgramObject_PF);
				System.out.println("RTR: Per Fragment Lighting Shader Program Linking Error: \n" + szInfoLog);
				szInfoLog = null;
				uninitialize();
				System.exit(1);
			} 
		}



		modelMatrixUniform_PF = GLES32.glGetUniformLocation(shaderProgramObject_PF, "u_model_matrix");
		viewMatrixUniform_PF = GLES32.glGetUniformLocation(shaderProgramObject_PF, "u_view_matrix");
		projectionMatrixUniform_PF = GLES32.glGetUniformLocation(shaderProgramObject_PF, "u_projection_matrix");
		la_Uniform_PF = GLES32.glGetUniformLocation(shaderProgramObject_PF, "u_la");
		ld_Uniform_PF = GLES32.glGetUniformLocation(shaderProgramObject_PF, "u_ld");
		ls_Uniform_PF = GLES32.glGetUniformLocation(shaderProgramObject_PF, "u_ls");
		lightPositionUniform_PF = GLES32.glGetUniformLocation(shaderProgramObject_PF, "u_light_position");

		ka_Uniform_PF = GLES32.glGetUniformLocation(shaderProgramObject_PF, "u_ka");
		kd_Uniform_PF = GLES32.glGetUniformLocation(shaderProgramObject_PF, "u_kd");
		ks_Uniform_PF = GLES32.glGetUniformLocation(shaderProgramObject_PF, "u_ks");
		shininessUniform_PF = GLES32.glGetUniformLocation(shaderProgramObject_PF, "u_shininess");
		LKeyPressUniform_PF = GLES32.glGetUniformLocation(shaderProgramObject_PF, "u_LKeyPress");







		Sphere sphere = new Sphere();
		float sphere_Position[] = new float[1146];
		float sphere_Normal[] = new float[1146];
		float sphere_TexCoord[] = new float[764];
		short sphere_Element[] = new short[2280];
		

		sphere.getSphereVertexData(sphere_Position, sphere_Normal, sphere_TexCoord, sphere_Element);
		numVertices = sphere.getNumberOfSphereVertices();
		numElements = sphere.getNumberOfSphereElements();


		GLES32.glGenVertexArrays(1, vao_Sphere, 0);
		GLES32.glBindVertexArray(vao_Sphere[0]);

			/********** Position **********/
			GLES32.glGenBuffers(1, vbo_Sphere_Position, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Sphere_Position[0]);

				ByteBuffer spherePosition_ByteBuffer = ByteBuffer.allocateDirect(sphere_Position.length * 4);
				spherePosition_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer spherePosition_FloatBuffer = spherePosition_ByteBuffer.asFloatBuffer();
				spherePosition_FloatBuffer.put(sphere_Position);
				spherePosition_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							sphere_Position.length * 4,
							spherePosition_FloatBuffer,
							GLES32.GL_STATIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);



			/********** Normal **********/ 
			GLES32.glGenBuffers(1, vbo_Sphere_Normal, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Sphere_Normal[0]);

				ByteBuffer sphereNormal_ByteBuffer = ByteBuffer.allocateDirect(sphere_Normal.length * 4);
				sphereNormal_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer sphereNormal_FloatBuffer = sphereNormal_ByteBuffer.asFloatBuffer();
				sphereNormal_FloatBuffer.put(sphere_Normal);
				sphereNormal_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							sphere_Normal.length * 4,
							sphereNormal_FloatBuffer,
							GLES32.GL_STATIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_NORMAL,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_NORMAL);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);


			/********** Elements **********/
			GLES32.glGenBuffers(1, vbo_Sphere_Element, 0);
			GLES32.glBindBuffer(GLES32.GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element[0]);

				ByteBuffer sphereElement_ByteBuffer = ByteBuffer.allocateDirect(sphere_Element.length * 4);
				sphereElement_ByteBuffer.order(ByteOrder.nativeOrder());
				ShortBuffer sphereElement_ShortBuffer = sphereElement_ByteBuffer.asShortBuffer();
				sphereElement_ShortBuffer.put(sphere_Element);
				sphereElement_ShortBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ELEMENT_ARRAY_BUFFER,
							sphere_Element.length * 4,
							sphereElement_ShortBuffer,
							GLES32.GL_STATIC_DRAW);
			GLES32.glBindBuffer(GLES32.GL_ELEMENT_ARRAY_BUFFER, 0);


		GLES32.glBindVertexArray(0);

		GLES32.glEnable(GLES32.GL_DEPTH_TEST);
		GLES32.glDepthFunc(GLES32.GL_LEQUAL);

		GLES32.glClearColor(0.250f, 0.250f, 0.250f, 0.0f);

	}

	private void uninitialize(){


		if(vbo_Sphere_Element[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Sphere_Element, 0);
			vbo_Sphere_Element[0] = 0;
		}

		if(vbo_Sphere_Normal[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Sphere_Normal, 0);
			vbo_Sphere_Normal[0] = 0;
		}

		if(vbo_Sphere_Position[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Sphere_Position, 0);
			vbo_Sphere_Position[0] = 0;
		}

		if(vao_Sphere[0] != 0){
			GLES32.glDeleteVertexArrays(1, vao_Sphere, 0);
			vao_Sphere[0] = 0;
		}


		if(shaderProgramObject_PF != 0){

			GLES32.glUseProgram(shaderProgramObject_PF);

				/*int iShaderCount[] = new int[1];
				int iShaderNo = 0;
				GLES32.glGetProgramiv(shaderProgramObject_PF, GLES32.GL_ATTACHED_SHADERS, iShaderCount, 0);
				System.out.println("RTR: ShaderCount: " + iShaderCount[0]);
				int iShaders[] = new int[iShaderCount[0]];
				GLES32.glGetAttachedShaders(shaderProgramObject_PF, iShaderCount[0], iShaderCount, 0, iShaders, 0);
				for(iShaderNo =0; iShaderNo < iShaderCount[0]; iShaderNo++){
					GLES32.glDetachShader(shaderProgramObject_PF, iShaders[iShaderNo]);
					GLES32.glDeleteShader(iShaders[iShaderNo]);
					iShaders[iShaderNo] = 0;
				}*/

				if(fragmentShaderObject_PF != 0){
					GLES32.glDetachShader(shaderProgramObject_PF, fragmentShaderObject_PF);
					GLES32.glDeleteShader(fragmentShaderObject_PF);
					fragmentShaderObject_PF= 0;
				}

				if(vertexShaderObject_PF != 0){
					GLES32.glDetachShader(shaderProgramObject_PF, vertexShaderObject_PF);
					GLES32.glDeleteShader(vertexShaderObject_PF);
					vertexShaderObject_PF = 0;
				}

			GLES32.glUseProgram(0);
			GLES32.glDeleteProgram(shaderProgramObject_PF);
			shaderProgramObject_PF = 0; 
		}


		if(shaderProgramObject_PV != 0){

			GLES32.glUseProgram(shaderProgramObject_PV);

				/*int iShaderCount[] = new int[1];
				int iShaderNo = 0;
				GLES32.glGetProgramiv(shaderProgramObject_PV, GLES32.GL_ATTACHED_SHADERS, iShaderCount, 0);
				System.out.println("RTR: ShaderCount: " + iShaderCount[0]);
				int iShaders[] = new int[iShaderCount[0]];
				GLES32.glGetAttachedShaders(shaderProgramObject_PV, iShaderCount[0], iShaderCount, 0, iShaders, 0);
				for(iShaderNo =0; iShaderNo < iShaderCount[0]; iShaderNo++){
					GLES32.glDetachShader(shaderProgramObject_PV, iShaders[iShaderNo]);
					GLES32.glDeleteShader(iShaders[iShaderNo]);
					iShaders[iShaderNo] = 0;
				}*/

				if(fragmentShaderObject_PV != 0){
					GLES32.glDetachShader(shaderProgramObject_PV, fragmentShaderObject_PV);
					GLES32.glDeleteShader(fragmentShaderObject_PV);
					fragmentShaderObject_PV= 0;
				}

				if(vertexShaderObject_PV != 0){
					GLES32.glDetachShader(shaderProgramObject_PV, vertexShaderObject_PV);
					GLES32.glDeleteShader(vertexShaderObject_PV);
					vertexShaderObject_PV = 0;
				}

			GLES32.glUseProgram(0);
			GLES32.glDeleteProgram(shaderProgramObject_PV);
			shaderProgramObject_PV = 0; 
		}
	}


	private void resize(int w, int h){
		if(h == 0)
			h = 1;


		viewPortWidth = w;
		viewPortHeight = h;

		if(iViewPortNo == 1)							/************ 1st SET ***********/
			GLES32.glViewport(150 + 0, 5 * h / 6, w / 6, h / 6);
		else if(iViewPortNo == 2)
			GLES32.glViewport(150 + 0, 4 * h / 6, w / 6,  h / 6);
		else if(iViewPortNo == 3)
			GLES32.glViewport(150 + 0, 3 * h / 6, w / 6, h / 6);
		else if(iViewPortNo == 4)
			GLES32.glViewport(150 + 0, 2 * h / 6, w / 6, h / 6);
		else if(iViewPortNo == 5)
			GLES32.glViewport(150 + 0, 1 * h / 6, w / 6, h / 6);
		else if(iViewPortNo == 6)
			GLES32.glViewport(150 + 0, 0, w / 6, h / 6);
		else if(iViewPortNo == 7)						/************ 2nd SET ***********/
			GLES32.glViewport(200 + 1 * w / 4, 5 * h / 6, w / 6, h / 6);
		else if(iViewPortNo == 8)
			GLES32.glViewport(200 + 1 * w / 4, 4 * h / 6, w / 6,  h / 6);
		else if(iViewPortNo == 9)
			GLES32.glViewport(200 + 1 * w / 4, 3 * h / 6, w / 6, h / 6);
		else if(iViewPortNo == 10)
			GLES32.glViewport(200 + 1 * w / 4, 2 * h / 6, w / 6, h / 6);
		else if(iViewPortNo == 11)
			GLES32.glViewport(200 + 1 * w / 4, 1 * h / 6, w / 6, h / 6);
		else if(iViewPortNo == 12)
			GLES32.glViewport(200 + 1 * w / 4, 0, w / 6, h / 6);
		else if(iViewPortNo == 13)						/************ 3rd SET ***********/
			GLES32.glViewport(200 + 2 * w / 4, 5 * h / 6, w / 6, h / 6);
		else if(iViewPortNo == 14)						
			GLES32.glViewport(200 + 2 * w / 4, 4 * h / 6, w / 6,  h / 6);
		else if(iViewPortNo == 15)
			GLES32.glViewport(200 + 2 * w / 4, 3 * h / 6, w / 6, h / 6);
		else if(iViewPortNo == 16)
			GLES32.glViewport(200 + 2 * w / 4, 2 * h / 6, w / 6, h / 6);
		else if(iViewPortNo == 17)
			GLES32.glViewport(200 + 2 * w / 4, 1 * h / 6, w / 6, h / 6);
		else if(iViewPortNo == 18)						
			GLES32.glViewport(200 + 2 * w / 4, 0, w / 6, h / 6);
		else if(iViewPortNo == 19)						/************ 4th SET ***********/
			GLES32.glViewport(200 + 3 * w / 4, 5 * h / 6, w / 6, h / 6);
		else if(iViewPortNo == 20)
			GLES32.glViewport(200 + 3 * w / 4, 4 * h / 6, w / 6,  h / 6);
		else if(iViewPortNo == 21)
			GLES32.glViewport(200 + 3 * w / 4, 3 * h / 6, w / 6, h / 6);
		else if(iViewPortNo == 22)
			GLES32.glViewport(200 + 3 * w / 4, 2 * h / 6, w / 6, h / 6);
		else if(iViewPortNo == 23)
			GLES32.glViewport(200 + 3 * w / 4, 1 * h / 6, w / 6, h / 6);
		else if(iViewPortNo == 24)
			GLES32.glViewport(200 + 3 * w / 4, 0, w / 6, h / 6);


		Matrix.setIdentityM(perspectiveProjectionMatrix, 0);
		Matrix.perspectiveM(perspectiveProjectionMatrix, 0,
						45.0f,
						(float)w / (float)h,
						0.1f,
						100.0f);


	}


	float translateMatrix[] = new float[4*4];
	float modelMatrix[] = new float[4*4];
	float viewMatrix[] = new float[4*4];


	private void display(){

		GLES32.glClear(GLES32.GL_COLOR_BUFFER_BIT | GLES32.GL_DEPTH_BUFFER_BIT);	

		if(iWhichLight == PER_VERTEX){
			
			GLES32.glUseProgram(shaderProgramObject_PV);
			
				draw24SpherePerVertex();

			GLES32.glUseProgram(0);
		}
		else{

			GLES32.glUseProgram(shaderProgramObject_PF);

				draw24SpherePerFragment();

			GLES32.glUseProgram(0);
		}

		requestRender();

	}


	private void draw24SpherePerVertex(){
			
			float materialAmbient[] = new float[4];
			float materialDiffuse[] = new float[4];
			float materialSpecular[] = new float[4];
			float materialShininess = 0.0f;

			for(int i = 1 ; i <= 24; i++){


				if(i == 1){
					materialAmbient[0] = 0.0215f;
					materialAmbient[1] = 0.1745f;
					materialAmbient[2] = 0.215f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.07568f;
					materialDiffuse[1] = 0.61424f;
					materialDiffuse[2] = 0.07568f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.633f;
					materialSpecular[1] = 0.727811f;
					materialSpecular[2] = 0.633f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.6f * 128;

				}
				else if(i == 2){
					materialAmbient[0] = 0.135f;
					materialAmbient[1] = 0.2225f;
					materialAmbient[2] = 0.1575f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.54f;
					materialDiffuse[1] = 0.89f;
					materialDiffuse[2] = 0.63f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.316228f;
					materialSpecular[1] = 0.316228f;
					materialSpecular[2] = 0.316228f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.1f * 128;
				}
				else if(i == 3){
					materialAmbient[0] = 0.05375f;
					materialAmbient[1] = 0.05f;
					materialAmbient[2] = 0.06625f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.18275f;
					materialDiffuse[1] = 0.17f;
					materialDiffuse[2] = 0.22525f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.332741f;
					materialSpecular[1] = 0.328634f;
					materialSpecular[2] = 0.346435f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.3f * 128;
				}
				else if(i == 4){
					materialAmbient[0] = 0.25f;
					materialAmbient[1] = 0.20725f;
					materialAmbient[2] = 0.20725f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 1.0f;
					materialDiffuse[1] = 0.829f;
					materialDiffuse[2] = 0.829f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.296648f;
					materialSpecular[1] = 0.296648f;
					materialSpecular[2] = 0.296648f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.088f * 128;
				}
				else if(i == 5){
					materialAmbient[0] = 0.1745f;
					materialAmbient[1] = 0.01175f;
					materialAmbient[2] = 0.01175f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.61424f;
					materialDiffuse[1] = 0.04136f;
					materialDiffuse[2] = 0.04136f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.727811f;
					materialSpecular[1] = 0.626959f;
					materialSpecular[2] = 0.626959f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.6f * 128;
				}
				else if(i == 6){
					materialAmbient[0] = 0.1f;
					materialAmbient[1] = 0.18725f;
					materialAmbient[2] = 0.1745f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.396f;
					materialDiffuse[1] = 0.74151f;
					materialDiffuse[2] = 0.69102f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.297254f;
					materialSpecular[1] = 0.30829f;
					materialSpecular[2] = 0.306678f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.1f * 128;
				}
				else if(i == 7){
					materialAmbient[0] = 0.329412f;
					materialAmbient[1] = 0.223529f;
					materialAmbient[2] = 0.027451f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.780392f;
					materialDiffuse[1] = 0.568627f;
					materialDiffuse[2] = 0.113725f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.992157f;
					materialSpecular[1] = 0.941176f;
					materialSpecular[2] = 0.807843f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.21794872f * 128;
				}
				else if(i == 8){
					materialAmbient[0] = 0.2125f;
					materialAmbient[1] = 0.1275f;
					materialAmbient[2] = 0.054f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.714f;
					materialDiffuse[1] = 0.4284f;
					materialDiffuse[2] = 0.18144f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.393548f;
					materialSpecular[1] = 0.271906f;
					materialSpecular[2] = 0.166721f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.2f * 128;
				}
				else if(i == 9){
					materialAmbient[0] = 0.25f;
					materialAmbient[1] = 0.25f;
					materialAmbient[2] = 0.25f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.4f;
					materialDiffuse[1] = 0.4f;
					materialDiffuse[2] = 0.4f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.774597f;
					materialSpecular[1] = 0.774597f;
					materialSpecular[2] = 0.774597f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.6f * 128;
				}
				else if(i == 10){
					materialAmbient[0] = 0.19125f;
					materialAmbient[1] = 0.0735f;
					materialAmbient[2] = 0.0225f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.7038f;
					materialDiffuse[1] = 0.27048f;
					materialDiffuse[2] = 0.0828f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.256777f;
					materialSpecular[1] = 0.137622f;
					materialSpecular[2] = 0.086014f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.1f * 128;
				}
				else if(i == 11){
					materialAmbient[0] = 0.24725f;
					materialAmbient[1] = 0.1995f;
					materialAmbient[2] = 0.0745f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.75164f;
					materialDiffuse[1] = 0.60648f;
					materialDiffuse[2] = 0.22648f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.628281f;
					materialSpecular[1] = 0.555802f;
					materialSpecular[2] = 0.366065f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.4f * 128;
				}
				else if(i == 12){
					materialAmbient[0] = 0.19225f;
					materialAmbient[1] = 0.19225f;
					materialAmbient[2] = 0.19225f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.50754f;
					materialDiffuse[1] = 0.50754f;
					materialDiffuse[2] = 0.50754f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.508273f;
					materialSpecular[1] = 0.508273f;
					materialSpecular[2] = 0.508273f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.4f * 128;
				}
				else if(i == 13){
					materialAmbient[0] = 0.0f;
					materialAmbient[1] = 0.0f;
					materialAmbient[2] = 0.0f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.01f;
					materialDiffuse[1] = 0.01f;
					materialDiffuse[2] = 0.01f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.5f;
					materialSpecular[1] = 0.5f;
					materialSpecular[2] = 0.5f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.25f * 128;
				}
				else if(i == 14){
					materialAmbient[0] = 0.0f;
					materialAmbient[1] = 0.1f;
					materialAmbient[2] = 0.06f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.0f;
					materialDiffuse[1] = 0.50980392f;
					materialDiffuse[2] = 0.52980392f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.50196078f;
					materialSpecular[1] = 0.50196078f;
					materialSpecular[2] = 0.50196078f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.25f * 128;
				}
				else if(i == 15){
					materialAmbient[0] = 0.0f;
					materialAmbient[1] = 0.0f;
					materialAmbient[2] = 0.0f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.1f;
					materialDiffuse[1] = 0.35f;
					materialDiffuse[2] = 0.1f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.45f;
					materialSpecular[1] = 0.55f;
					materialSpecular[2] = 0.45f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.25f * 128;
				}
				else if(i == 16){
					materialAmbient[0] = 0.0f;
					materialAmbient[1] = 0.0f;
					materialAmbient[2] = 0.0f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.5f;
					materialDiffuse[1] = 0.0f;
					materialDiffuse[2] = 0.0f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.7f;
					materialSpecular[1] = 0.6f;
					materialSpecular[2] = 0.6f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.25f * 128;
				}
				else if(i == 17){
					materialAmbient[0] = 0.0f;
					materialAmbient[1] = 0.0f;
					materialAmbient[2] = 0.0f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.55f;
					materialDiffuse[1] = 0.55f;
					materialDiffuse[2] = 0.55f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.70f;
					materialSpecular[1] = 0.70f;
					materialSpecular[2] = 0.70f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.25f * 128;
				}
				else if(i == 18){
					materialAmbient[0] = 0.0f;
					materialAmbient[1] = 0.0f;
					materialAmbient[2] = 0.0f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.5f;
					materialDiffuse[1] = 0.5f;
					materialDiffuse[2] = 0.0f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.60f;
					materialSpecular[1] = 0.60f;
					materialSpecular[2] = 0.50f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.25f * 128;
				}
				else if(i == 19){
					materialAmbient[0] = 0.02f;
					materialAmbient[1] = 0.02f;
					materialAmbient[2] = 0.02f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.01f;
					materialDiffuse[1] = 0.01f;
					materialDiffuse[2] = 0.01f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.4f;
					materialSpecular[1] = 0.4f;
					materialSpecular[2] = 0.4f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.078125f * 128;
				}
				else if(i == 20){
					materialAmbient[0] = 0.0f;
					materialAmbient[1] = 0.05f;
					materialAmbient[2] = 0.05f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.4f;
					materialDiffuse[1] = 0.5f;
					materialDiffuse[2] = 0.5f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.04f;
					materialSpecular[1] = 0.7f;
					materialSpecular[2] = 0.7f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.078125f * 128;
				}
				else if(i == 21){
					materialAmbient[0] = 0.0f;
					materialAmbient[1] = 0.05f;
					materialAmbient[2] = 0.0f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.4f;
					materialDiffuse[1] = 0.5f;
					materialDiffuse[2] = 0.4f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.04f;
					materialSpecular[1] = 0.7f;
					materialSpecular[2] = 0.04f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.078125f * 128;
				}
				else if(i == 22){
					materialAmbient[0] = 0.05f;
					materialAmbient[1] = 0.0f;
					materialAmbient[2] = 0.0f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.5f;
					materialDiffuse[1] = 0.4f;
					materialDiffuse[2] = 0.4f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.70f;
					materialSpecular[1] = 0.04f;
					materialSpecular[2] = 0.04f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.078125f * 128;
				}
				else if(i == 23){
					materialAmbient[0] = 0.05f;
					materialAmbient[1] = 0.05f;
					materialAmbient[2] = 0.05f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.5f;
					materialDiffuse[1] = 0.5f;
					materialDiffuse[2] = 0.5f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.70f;
					materialSpecular[1] = 0.70f;
					materialSpecular[2] = 0.70f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.078125f * 128;
				}
				else if(i == 24){
					materialAmbient[0] = 0.05f;
					materialAmbient[1] = 0.05f;
					materialAmbient[2] = 0.0f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.5f;
					materialDiffuse[1] = 0.5f;
					materialDiffuse[2] = 0.4f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.70f;
					materialSpecular[1] = 0.70f;
					materialSpecular[2] = 0.04f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.078125f * 128;
				}


				iViewPortNo = i;
				resize(viewPortWidth, viewPortHeight);

				Matrix.setIdentityM(translateMatrix, 0);
				Matrix.setIdentityM(modelMatrix, 0);
				Matrix.setIdentityM(viewMatrix, 0);

				Matrix.translateM(translateMatrix, 0, 0.0f, 0.0f, -1.50f);
				Matrix.multiplyMM(modelMatrix, 0, modelMatrix, 0, translateMatrix, 0);

				GLES32.glUniformMatrix4fv(modelMatrixUniform_PV, 1, false, modelMatrix, 0);
				GLES32.glUniformMatrix4fv(viewMatrixUniform_PV, 1, false, viewMatrix, 0);
				GLES32.glUniformMatrix4fv(projectionMatrixUniform_PV, 1, false, perspectiveProjectionMatrix, 0);
				
				
				if(bLights == true){

						if(iWhichRotation == X_ROT)
							rotateX(angle_X);
						else if(iWhichRotation == Y_ROT)
							rotateY(angle_Y);
						else if(iWhichRotation == Z_ROT)
							rotateZ(angle_Z);


						GLES32.glUniform1i(LKeyPressUniform_PV, 1);

						GLES32.glUniform3fv(la_Uniform_PV, 1, lightAmbient, 0);
						GLES32.glUniform3fv(ld_Uniform_PV, 1, lightDiffuse, 0);
						GLES32.glUniform3fv(ls_Uniform_PV, 1, lightSpecular, 0);
						GLES32.glUniform4fv(lightPositionUniform_PV, 1, lightPosition, 0);

						GLES32.glUniform3fv(ka_Uniform_PV, 1, materialAmbient, 0);
						GLES32.glUniform3fv(kd_Uniform_PV, 1, materialDiffuse, 0);
						GLES32.glUniform3fv(ks_Uniform_PV, 1, materialSpecular, 0);
						GLES32.glUniform1f(shininessUniform_PV, materialShininess);

				}
				else
					GLES32.glUniform1i(LKeyPressUniform_PV, 0);

				GLES32.glBindVertexArray(vao_Sphere[0]);

				GLES32.glBindBuffer(GLES32.GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element[0]);
				GLES32.glDrawElements(GLES32.GL_TRIANGLES, numElements, GLES32.GL_UNSIGNED_SHORT, 0);
				GLES32.glBindBuffer(GLES32.GL_ELEMENT_ARRAY_BUFFER, 0);

				GLES32.glBindVertexArray(0);

			}


	}


	private void draw24SpherePerFragment(){

		float materialAmbient[] = new float[4];
		float materialDiffuse[] = new float[4];
		float materialSpecular[] = new float[4];
		float materialShininess = 0.0f;





		for(int i = 1; i <= 24; i++){

			if(i == 1){
					materialAmbient[0] = 0.0215f;
					materialAmbient[1] = 0.1745f;
					materialAmbient[2] = 0.215f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.07568f;
					materialDiffuse[1] = 0.61424f;
					materialDiffuse[2] = 0.07568f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.633f;
					materialSpecular[1] = 0.727811f;
					materialSpecular[2] = 0.633f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.6f * 128;

				}
				else if(i == 2){
					materialAmbient[0] = 0.135f;
					materialAmbient[1] = 0.2225f;
					materialAmbient[2] = 0.1575f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.54f;
					materialDiffuse[1] = 0.89f;
					materialDiffuse[2] = 0.63f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.316228f;
					materialSpecular[1] = 0.316228f;
					materialSpecular[2] = 0.316228f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.1f * 128;
				}
				else if(i == 3){
					materialAmbient[0] = 0.05375f;
					materialAmbient[1] = 0.05f;
					materialAmbient[2] = 0.06625f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.18275f;
					materialDiffuse[1] = 0.17f;
					materialDiffuse[2] = 0.22525f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.332741f;
					materialSpecular[1] = 0.328634f;
					materialSpecular[2] = 0.346435f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.3f * 128;
				}
				else if(i == 4){
					materialAmbient[0] = 0.25f;
					materialAmbient[1] = 0.20725f;
					materialAmbient[2] = 0.20725f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 1.0f;
					materialDiffuse[1] = 0.829f;
					materialDiffuse[2] = 0.829f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.296648f;
					materialSpecular[1] = 0.296648f;
					materialSpecular[2] = 0.296648f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.088f * 128;
				}
				else if(i == 5){
					materialAmbient[0] = 0.1745f;
					materialAmbient[1] = 0.01175f;
					materialAmbient[2] = 0.01175f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.61424f;
					materialDiffuse[1] = 0.04136f;
					materialDiffuse[2] = 0.04136f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.727811f;
					materialSpecular[1] = 0.626959f;
					materialSpecular[2] = 0.626959f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.6f * 128;
				}
				else if(i == 6){
					materialAmbient[0] = 0.1f;
					materialAmbient[1] = 0.18725f;
					materialAmbient[2] = 0.1745f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.396f;
					materialDiffuse[1] = 0.74151f;
					materialDiffuse[2] = 0.69102f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.297254f;
					materialSpecular[1] = 0.30829f;
					materialSpecular[2] = 0.306678f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.1f * 128;
				}
				else if(i == 7){
					materialAmbient[0] = 0.329412f;
					materialAmbient[1] = 0.223529f;
					materialAmbient[2] = 0.027451f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.780392f;
					materialDiffuse[1] = 0.568627f;
					materialDiffuse[2] = 0.113725f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.992157f;
					materialSpecular[1] = 0.941176f;
					materialSpecular[2] = 0.807843f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.21794872f * 128;
				}
				else if(i == 8){
					materialAmbient[0] = 0.2125f;
					materialAmbient[1] = 0.1275f;
					materialAmbient[2] = 0.054f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.714f;
					materialDiffuse[1] = 0.4284f;
					materialDiffuse[2] = 0.18144f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.393548f;
					materialSpecular[1] = 0.271906f;
					materialSpecular[2] = 0.166721f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.2f * 128;
				}
				else if(i == 9){
					materialAmbient[0] = 0.25f;
					materialAmbient[1] = 0.25f;
					materialAmbient[2] = 0.25f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.4f;
					materialDiffuse[1] = 0.4f;
					materialDiffuse[2] = 0.4f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.774597f;
					materialSpecular[1] = 0.774597f;
					materialSpecular[2] = 0.774597f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.6f * 128;
				}
				else if(i == 10){
					materialAmbient[0] = 0.19125f;
					materialAmbient[1] = 0.0735f;
					materialAmbient[2] = 0.0225f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.7038f;
					materialDiffuse[1] = 0.27048f;
					materialDiffuse[2] = 0.0828f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.256777f;
					materialSpecular[1] = 0.137622f;
					materialSpecular[2] = 0.086014f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.1f * 128;
				}
				else if(i == 11){
					materialAmbient[0] = 0.24725f;
					materialAmbient[1] = 0.1995f;
					materialAmbient[2] = 0.0745f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.75164f;
					materialDiffuse[1] = 0.60648f;
					materialDiffuse[2] = 0.22648f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.628281f;
					materialSpecular[1] = 0.555802f;
					materialSpecular[2] = 0.366065f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.4f * 128;
				}
				else if(i == 12){
					materialAmbient[0] = 0.19225f;
					materialAmbient[1] = 0.19225f;
					materialAmbient[2] = 0.19225f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.50754f;
					materialDiffuse[1] = 0.50754f;
					materialDiffuse[2] = 0.50754f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.508273f;
					materialSpecular[1] = 0.508273f;
					materialSpecular[2] = 0.508273f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.4f * 128;
				}
				else if(i == 13){
					materialAmbient[0] = 0.0f;
					materialAmbient[1] = 0.0f;
					materialAmbient[2] = 0.0f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.01f;
					materialDiffuse[1] = 0.01f;
					materialDiffuse[2] = 0.01f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.5f;
					materialSpecular[1] = 0.5f;
					materialSpecular[2] = 0.5f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.25f * 128;
				}
				else if(i == 14){
					materialAmbient[0] = 0.0f;
					materialAmbient[1] = 0.1f;
					materialAmbient[2] = 0.06f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.0f;
					materialDiffuse[1] = 0.50980392f;
					materialDiffuse[2] = 0.52980392f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.50196078f;
					materialSpecular[1] = 0.50196078f;
					materialSpecular[2] = 0.50196078f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.25f * 128;
				}
				else if(i == 15){
					materialAmbient[0] = 0.0f;
					materialAmbient[1] = 0.0f;
					materialAmbient[2] = 0.0f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.1f;
					materialDiffuse[1] = 0.35f;
					materialDiffuse[2] = 0.1f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.45f;
					materialSpecular[1] = 0.55f;
					materialSpecular[2] = 0.45f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.25f * 128;
				}
				else if(i == 16){
					materialAmbient[0] = 0.0f;
					materialAmbient[1] = 0.0f;
					materialAmbient[2] = 0.0f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.5f;
					materialDiffuse[1] = 0.0f;
					materialDiffuse[2] = 0.0f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.7f;
					materialSpecular[1] = 0.6f;
					materialSpecular[2] = 0.6f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.25f * 128;
				}
				else if(i == 17){
					materialAmbient[0] = 0.0f;
					materialAmbient[1] = 0.0f;
					materialAmbient[2] = 0.0f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.55f;
					materialDiffuse[1] = 0.55f;
					materialDiffuse[2] = 0.55f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.70f;
					materialSpecular[1] = 0.70f;
					materialSpecular[2] = 0.70f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.25f * 128;
				}
				else if(i == 18){
					materialAmbient[0] = 0.0f;
					materialAmbient[1] = 0.0f;
					materialAmbient[2] = 0.0f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.5f;
					materialDiffuse[1] = 0.5f;
					materialDiffuse[2] = 0.0f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.60f;
					materialSpecular[1] = 0.60f;
					materialSpecular[2] = 0.50f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.25f * 128;
				}
				else if(i == 19){
					materialAmbient[0] = 0.02f;
					materialAmbient[1] = 0.02f;
					materialAmbient[2] = 0.02f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.01f;
					materialDiffuse[1] = 0.01f;
					materialDiffuse[2] = 0.01f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.4f;
					materialSpecular[1] = 0.4f;
					materialSpecular[2] = 0.4f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.078125f * 128;
				}
				else if(i == 20){
					materialAmbient[0] = 0.0f;
					materialAmbient[1] = 0.05f;
					materialAmbient[2] = 0.05f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.4f;
					materialDiffuse[1] = 0.5f;
					materialDiffuse[2] = 0.5f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.04f;
					materialSpecular[1] = 0.7f;
					materialSpecular[2] = 0.7f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.078125f * 128;
				}
				else if(i == 21){
					materialAmbient[0] = 0.0f;
					materialAmbient[1] = 0.05f;
					materialAmbient[2] = 0.0f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.4f;
					materialDiffuse[1] = 0.5f;
					materialDiffuse[2] = 0.4f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.04f;
					materialSpecular[1] = 0.7f;
					materialSpecular[2] = 0.04f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.078125f * 128;
				}
				else if(i == 22){
					materialAmbient[0] = 0.05f;
					materialAmbient[1] = 0.0f;
					materialAmbient[2] = 0.0f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.5f;
					materialDiffuse[1] = 0.4f;
					materialDiffuse[2] = 0.4f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.70f;
					materialSpecular[1] = 0.04f;
					materialSpecular[2] = 0.04f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.078125f * 128;
				}
				else if(i == 23){
					materialAmbient[0] = 0.05f;
					materialAmbient[1] = 0.05f;
					materialAmbient[2] = 0.05f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.5f;
					materialDiffuse[1] = 0.5f;
					materialDiffuse[2] = 0.5f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.70f;
					materialSpecular[1] = 0.70f;
					materialSpecular[2] = 0.70f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.078125f * 128;
				}
				else if(i == 24){
					materialAmbient[0] = 0.05f;
					materialAmbient[1] = 0.05f;
					materialAmbient[2] = 0.0f;
					materialAmbient[3] = 1.0f;

					materialDiffuse[0] = 0.5f;
					materialDiffuse[1] = 0.5f;
					materialDiffuse[2] = 0.4f;
					materialDiffuse[3] = 1.0f;

					materialSpecular[0] = 0.70f;
					materialSpecular[1] = 0.70f;
					materialSpecular[2] = 0.04f;
					materialSpecular[3] = 1.0f;

					materialShininess = 0.078125f * 128;
				}


				iViewPortNo = i;
				resize(viewPortWidth, viewPortHeight);

			Matrix.setIdentityM(translateMatrix, 0);
			Matrix.setIdentityM(modelMatrix, 0);
			Matrix.setIdentityM(viewMatrix, 0);

			Matrix.translateM(translateMatrix, 0, 0.0f, 0.0f, -1.50f);
			Matrix.multiplyMM(modelMatrix, 0, modelMatrix, 0, translateMatrix, 0);

			GLES32.glUniformMatrix4fv(modelMatrixUniform_PF, 1, false, modelMatrix, 0);
			GLES32.glUniformMatrix4fv(viewMatrixUniform_PF, 1, false, viewMatrix, 0);
			GLES32.glUniformMatrix4fv(projectionMatrixUniform_PF, 1, false, perspectiveProjectionMatrix, 0);
	
			if(bLights == true){

					if(iWhichRotation == X_ROT)
						rotateX(angle_X);
					else if(iWhichRotation == Y_ROT)
						rotateY(angle_Y);
					else if(iWhichRotation == Z_ROT)
						rotateZ(angle_Z);

					GLES32.glUniform1i(LKeyPressUniform_PF, 1);

					GLES32.glUniform3fv(la_Uniform_PF, 1, lightAmbient, 0);
					GLES32.glUniform3fv(ld_Uniform_PF, 1, lightDiffuse, 0);
					GLES32.glUniform3fv(ls_Uniform_PF, 1, lightSpecular, 0);
					GLES32.glUniform4fv(lightPositionUniform_PF, 1, lightPosition, 0);

					GLES32.glUniform3fv(ka_Uniform_PF, 1, materialAmbient, 0);
					GLES32.glUniform3fv(kd_Uniform_PF, 1, materialDiffuse, 0);
					GLES32.glUniform3fv(ks_Uniform_PF, 1, materialSpecular, 0);
					GLES32.glUniform1f(shininessUniform_PF, materialShininess);

			}
			else
				GLES32.glUniform1i(LKeyPressUniform_PF, 0);

			GLES32.glBindVertexArray(vao_Sphere[0]);

			GLES32.glBindBuffer(GLES32.GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element[0]);
			GLES32.glDrawElements(GLES32.GL_TRIANGLES, numElements, GLES32.GL_UNSIGNED_SHORT, 0);
			GLES32.glBindBuffer(GLES32.GL_ELEMENT_ARRAY_BUFFER, 0);

			GLES32.glBindVertexArray(0);
		}

	}


	private void rotateX(float angle){
		lightPosition[1] = (float)(5.0f * Math.sin(angle));
		lightPosition[2] = (float)(5.0f * Math.cos(angle));
	}

	private void rotateY(float angle){
		lightPosition[0] = (float)(5.0f * Math.sin(angle));
		lightPosition[2] = (float)(5.0f * Math.cos(angle));
	}

	private void rotateZ(float angle){
		lightPosition[0] = (float)(5.0f * Math.cos(angle));
		lightPosition[1] = (float)(5.0f * Math.sin(angle));
	}


	private void update(){
		if(iWhichRotation == X_ROT)
			angle_X = angle_X + 0.02f;
		else if(iWhichRotation == Y_ROT)
			angle_Y = angle_Y + 0.02f;
		else if(iWhichRotation == Z_ROT)
			angle_Z = angle_Z + 0.02f;

		if(angle_X > 360.0f)
			angle_X = 0.0f;

		if(angle_Y > 360.0f)
			angle_Y = 0.0f;

		if(angle_Z > 360.0f)
			angle_Z = 0.0f;
	}

}
