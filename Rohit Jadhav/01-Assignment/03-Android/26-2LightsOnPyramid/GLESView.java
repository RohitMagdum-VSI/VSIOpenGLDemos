package com.rohit_r_jadhav.lights_on_pyramid;

import android.opengl.GLSurfaceView;
import android.opengl.GLES32;
import android.opengl.Matrix;

import javax.microedition.khronos.opengles.GL10;
import javax.microedition.khronos.egl.EGLConfig;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import android.view.MotionEvent;
import android.view.GestureDetector;
import android.view.GestureDetector.OnDoubleTapListener;
import android.view.GestureDetector.OnGestureListener;
import android.content.Context;

public class GLESView extends GLSurfaceView implements GLSurfaceView.Renderer, OnDoubleTapListener, OnGestureListener{

	final int PER_VERTEX = 1;
	final int PER_FRAGMENT = 2;

	GestureDetector gestureDetector;
	Context context;
	boolean bLights = false;
	int iWhichLight = PER_VERTEX;


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


	/*********** Methods from OnDoubleTapListener **********/
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
		else
			bLights = false;
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
	public boolean onScroll(MotionEvent e1, MotionEvent e2, float distanceX,  float distanceY){
		uninitialize();
		System.exit(0);
		return(true);
	}



	/********** Methods from GLSurfaceView.Renderer **********/
	@Override
	public void onSurfaceCreated(GL10 gl, EGLConfig config){

		String version = gl.glGetString(GL10.GL_VERSION);
		String glsl_version = gl.glGetString(GLES32.GL_SHADING_LANGUAGE_VERSION);
		String vendor = gl.glGetString(GL10.GL_VENDOR);
		String renderer = gl.glGetString(GL10.GL_RENDERER);

		System.out.println("RTR: OpenGL Version: " + version);
		System.out.println("RTR: OpenGLSL Version: " + glsl_version);
		System.out.println("RTR: Vendor: " + vendor);
		System.out.println("RTR: Renderer: "+ renderer);

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




	//For Pyramid
	private int vao_Pyramid[] = new int[1];
	private int vbo_Pyramid_Position[] = new int[1];
	private int vbo_Pyramid_Normal[] = new int[1];
	private float angle_pyramid = 0.0f;

	//For Shader
	private int vertexShaderObject_PV;
	private int fragmentShaderObject_PV;
	private int shaderProgramObject_PV;

	private int vertexShaderObject_PF;
	private int fragmentShaderObject_PF;
	private int shaderProgramObject_PF;


	//For Projection
	float perspectiveProjectionMatrix[] = new float[4*4];


	//Per Vertex uniform
	private int modelMatrixUniform_PV;
	private int viewMatrixUniform_PV;
	private int projectionMatrixUniform_PV;

	private int red_la_Uniform_PV;
	private int red_ld_Uniform_PV;
	private int red_ls_Uniform_PV;
	private int red_lightPositionUniform_PV;

	private int blue_la_Uniform_PV;
	private int blue_ld_Uniform_PV;
	private int blue_ls_Uniform_PV;
	private int blue_lightPositionUniform_PV;

	private int ka_Uniform_PV;
	private int kd_Uniform_PV;
	private int ks_Uniform_PV;
	private int shininessUniform_PV;
	private int LKeyPressUniform_PV;

	

	//Per Fragment uniform
	private int modelMatrixUniform_PF;
	private int viewMatrixUniform_PF;
	private int projectionMatrixUniform_PF;

	private int red_la_Uniform_PF;
	private int red_ld_Uniform_PF;
	private int red_ls_Uniform_PF;
	private int red_lightPositionUniform_PF;

	private int blue_la_Uniform_PF;
	private int blue_ld_Uniform_PF;
	private int blue_ls_Uniform_PF;
	private int blue_lightPositionUniform_PF;

	private int ka_Uniform_PF;
	private int kd_Uniform_PF;
	private int ks_Uniform_PF;
	private int shininessUniform_PF;
	private int LKeyPressUniform_PF;


	//For Red Lights
	float lightAmbient_Red[] = new float[]{0.0f, 0.0f, 0.0f, 0.0f};
	float lightDiffuse_Red[] = new float[] {1.0f, 0.0f, 0.0f, 1.0f};
	float lightSpecular_Red[] = new float[]{1.0f, 0.0f, 0.0f, 1.0f};
	float lightPosition_Red[] = new float[]{-2.0f, 0.0f, 0.0f, 1.0f};

	//For Blue Lights
	float lightAmbient_Blue[] = new float[]{0.0f, 0.0f, 0.0f, 0.0f};
	float lightDiffuse_Blue[] = new float[] {0.0f, 0.0f, 1.0f, 1.0f};
	float lightSpecular_Blue[] = new float[]{0.0f, 0.0f, 1.0f, 1.0f};
	float lightPosition_Blue[] = new float[]{2.0f, 0.0f, 0.0f, 1.0f};



	//For Material
	float materialAmbient[] = new float[]{0.0f, 0.0f, 0.0f, 0.0f};
	float materialDiffuse[] = new float[] {1.0f, 1.0f, 1.0f, 1.0f};
	float materialSpecular[] = new float[] {1.0f, 1.0f, 1.0f, 1.0f};
	float materialShininess = 128.0f;

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

				"uniform vec3 u_Red_la;" +
				"uniform vec3 u_Red_ld;" +
				"uniform vec3 u_Red_ls;" +
				"uniform vec4 u_Red_light_position;" +

				"uniform vec3 u_Blue_la;" +
				"uniform vec3 u_Blue_ld;" +
				"uniform vec3 u_Blue_ls;" +
				"uniform vec4 u_Blue_light_position;" +

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
						"mat3 normalMatrix = mat3(u_view_matrix * u_model_matrix);" +
						"vec3 tNormal = normalize(vec3(normalMatrix * vNormal));" +

						"vec3 lightDirection_Red = normalize(vec3(u_Red_light_position - eyeCoordinate));" +
						"float SRed_Dot_N = max(dot(lightDirection_Red, tNormal), 0.0);" +

						"vec3 lightDirection_Blue = normalize(vec3(u_Blue_light_position - eyeCoordinate));" +
						"float SBlue_Dot_N = max(dot(lightDirection_Blue, tNormal), 0.0);" +

						"vec3 viewer = normalize(vec3(-eyeCoordinate.xyz));" +
						
						"vec3 reflection_Red = reflect(-lightDirection_Red, tNormal);" +
						"float RRed_Dot_V = max(dot(reflection_Red, viewer), 0.0);" +

						"vec3 reflection_Blue = reflect(-lightDirection_Blue, tNormal);" +
						"float RBlue_Dot_V = max(dot(reflection_Blue, viewer), 0.0);" +

						"vec3 ambient_Red = u_Red_la * u_ka;" +
						"vec3 diffuse_Red = u_Red_ld * u_kd * SRed_Dot_N;"+
						"vec3 specular_Red = u_Red_ls * u_ks * pow(RRed_Dot_V, u_shininess);" +
						"vec3 Red_Light = ambient_Red + diffuse_Red + specular_Red;" +

						"vec3 ambient_Blue = u_Blue_la * u_ka;" +
						"vec3 diffuse_Blue = u_Blue_ld * u_kd * SBlue_Dot_N;"+
						"vec3 specular_Blue = u_Blue_ls * u_ks * pow(RBlue_Dot_V, u_shininess);" +
						"vec3 Blue_Light = ambient_Blue + diffuse_Blue + specular_Blue;"+

						"phongLight = Red_Light + Blue_Light;" +	
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

		red_la_Uniform_PV = GLES32.glGetUniformLocation(shaderProgramObject_PV, "u_Red_la");
		red_ld_Uniform_PV = GLES32.glGetUniformLocation(shaderProgramObject_PV, "u_Red_ld");
		red_ls_Uniform_PV = GLES32.glGetUniformLocation(shaderProgramObject_PV, "u_Red_ls");
		red_lightPositionUniform_PV = GLES32.glGetUniformLocation(shaderProgramObject_PV, "u_Red_light_position");


		blue_la_Uniform_PV = GLES32.glGetUniformLocation(shaderProgramObject_PV, "u_Blue_la");
		blue_ld_Uniform_PV = GLES32.glGetUniformLocation(shaderProgramObject_PV, "u_Blue_ld");
		blue_ls_Uniform_PV = GLES32.glGetUniformLocation(shaderProgramObject_PV, "u_Blue_ls");
		blue_lightPositionUniform_PV = GLES32.glGetUniformLocation(shaderProgramObject_PV, "u_Blue_light_position");


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
				
				"uniform vec4 u_Red_light_position;" +
				"uniform vec4 u_Blue_light_position;" +
				
				"out vec3 Red_lightDirection_VS;" +
				"out vec3 Blue_lightDirection_VS;" +
				
				"out vec3 tNormal_VS;" +
				"out vec3 viewer_VS;" +
				"void main(void)" +
				"{" +
					"vec4 eyeCoordinate = u_view_matrix * u_model_matrix * vPosition;" +

					"Red_lightDirection_VS = vec3(u_Red_light_position - eyeCoordinate);" +
					"Blue_lightDirection_VS = vec3(u_Blue_light_position - eyeCoordinate);" +
					
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
				
				"in vec3 Red_lightDirection_VS;" +
				"in vec3 Blue_lightDirection_VS;" +

				"in vec3 tNormal_VS;" +
				"in vec3 viewer_VS;" +

				"uniform vec3 u_Red_la;" +
				"uniform vec3 u_Red_ld;" +
				"uniform vec3 u_Red_ls;" +

				"uniform vec3 u_Blue_la;" +
				"uniform vec3 u_Blue_ld;" +
				"uniform vec3 u_Blue_ls;" +

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
						"vec3 normalizeRedLightDirection = normalize(Red_lightDirection_VS);" +
						"vec3 normalizeBlueLightDirection = normalize(Blue_lightDirection_VS);" +

						"vec3 normalizeTNormal = normalize(tNormal_VS);" +
						"float SRed_Dot_N = max(dot(normalizeRedLightDirection, normalizeTNormal), 0.0);" +
						"float SBlue_Dot_N = max(dot(normalizeBlueLightDirection, normalizeTNormal), 0.0);" +

						"vec3 normalizeViewer = normalize(viewer_VS);" +

						"vec3 Red_Reflection = reflect(-normalizeRedLightDirection, normalizeTNormal);" +
						"float RRed_Dot_V = max(dot(Red_Reflection, normalizeViewer), 0.0);" +

						"vec3 Blue_Reflection = reflect(-normalizeBlueLightDirection, normalizeTNormal);" +
						"float RBlue_Dot_V = max(dot(Blue_Reflection, normalizeViewer), 0.0);" +


						"vec3 ambient_Red = u_Red_la * u_ka;" +
						"vec3 diffuse_Red = u_Red_ld * u_kd * SRed_Dot_N;"+
						"vec3 specular_Red = u_Red_ls * u_ks * pow(RRed_Dot_V, u_shininess);" +
						"vec3 Red_Light = ambient_Red + diffuse_Red + specular_Red;" +

						"vec3 ambient_Blue = u_Blue_la * u_ka;" +
						"vec3 diffuse_Blue = u_Blue_ld * u_kd * SBlue_Dot_N;"+
						"vec3 specular_Blue = u_Blue_ls * u_ks * pow(RBlue_Dot_V, u_shininess);" +
						"vec3 Blue_Light = ambient_Blue + diffuse_Blue + specular_Blue;"+

						"phongLight = Red_Light + Blue_Light;" +
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
		
		red_la_Uniform_PF = GLES32.glGetUniformLocation(shaderProgramObject_PF, "u_Red_la");
		red_ld_Uniform_PF = GLES32.glGetUniformLocation(shaderProgramObject_PF, "u_Red_ld");
		red_ls_Uniform_PF = GLES32.glGetUniformLocation(shaderProgramObject_PF, "u_Red_ls");
		red_lightPositionUniform_PF = GLES32.glGetUniformLocation(shaderProgramObject_PF, "u_Red_light_position");


		blue_la_Uniform_PF = GLES32.glGetUniformLocation(shaderProgramObject_PF, "u_Blue_la");
		blue_ld_Uniform_PF = GLES32.glGetUniformLocation(shaderProgramObject_PF, "u_Blue_ld");
		blue_ls_Uniform_PF = GLES32.glGetUniformLocation(shaderProgramObject_PF, "u_Blue_ls");
		blue_lightPositionUniform_PF = GLES32.glGetUniformLocation(shaderProgramObject_PF, "u_Blue_light_position");



		ka_Uniform_PF = GLES32.glGetUniformLocation(shaderProgramObject_PF, "u_ka");
		kd_Uniform_PF = GLES32.glGetUniformLocation(shaderProgramObject_PF, "u_kd");
		ks_Uniform_PF = GLES32.glGetUniformLocation(shaderProgramObject_PF, "u_ks");
		shininessUniform_PF = GLES32.glGetUniformLocation(shaderProgramObject_PF, "u_shininess");
		LKeyPressUniform_PF = GLES32.glGetUniformLocation(shaderProgramObject_PF, "u_LKeyPress");


		/********** Positions **********/
		float pyramid_Position[] = new float[]{
			//Face
			0.0f, 1.0f, 0.0f,
			-1.0f, -1.0f, 1.0f,
			1.0f, -1.0f, 1.0f,
			//Right
			0.0f, 1.0f, 0.0f,
			1.0f, -1.0f, 1.0f,
			1.0f, -1.0f, -1.0f,
			//Back
			0.0f, 1.0f, 0.0f,
			1.0f, -1.0f, -1.0f,
			-1.0f, -1.0f, -1.0f,
			//Left
			0.0f, 1.0f, 0.0f,
			-1.0f, -1.0f, -1.0f,
			-1.0f, -1.0f, 1.0f
		};


		float pyramid_Normal[] = new float[]{
			//Face
			0.0f, 0.447214f, 0.894427f,
			0.0f, 0.447214f, 0.894427f,
			0.0f, 0.447214f, 0.894427f,
			
			//Right
			0.894427f, 0.447214f, 0.0f,
			0.894427f, 0.447214f, 0.0f,
			0.894427f, 0.447214f, 0.0f,

			//Back
			0.0f, 0.447214f, -0.894427f,
			0.0f, 0.447214f, -0.894427f,
			0.0f, 0.447214f, -0.894427f,

			//Left
			-0.894427f, 0.447214f, 0.0f,
			-0.894427f, 0.447214f, 0.0f,
			-0.894427f, 0.447214f, 0.0f,

		};


		/********** Pyramid **********/
		GLES32.glGenVertexArrays(1, vao_Pyramid, 0);
		GLES32.glBindVertexArray(vao_Pyramid[0]);

			/********** Position **********/
			GLES32.glGenBuffers(1, vbo_Pyramid_Position, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Pyramid_Position[0]);

				/********** For glBufferData **********/
				ByteBuffer Pyramid_positionByteBuffer = ByteBuffer.allocateDirect(pyramid_Position.length * 4);
				Pyramid_positionByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer Pyramid_positionBuffer = Pyramid_positionByteBuffer.asFloatBuffer();
				Pyramid_positionBuffer.put(pyramid_Position);
				Pyramid_positionBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							pyramid_Position.length * 4,
							Pyramid_positionBuffer,
							GLES32.GL_STATIC_DRAW);

			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,	
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);

			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);


			/********** Normal **********/
			GLES32.glGenBuffers(1, vbo_Pyramid_Normal, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Pyramid_Normal[0]);

				ByteBuffer pyramidNormal_ByteBuffer = ByteBuffer.allocateDirect(pyramid_Normal.length * 4);
				pyramidNormal_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer pyramidNormal_FloatBuffer = pyramidNormal_ByteBuffer.asFloatBuffer();
				pyramidNormal_FloatBuffer.put(pyramid_Normal);
				pyramidNormal_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER, 
							pyramid_Normal.length * 4,
							pyramidNormal_FloatBuffer, 
							GLES32.GL_STATIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_NORMAL, 
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_NORMAL);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glBindVertexArray(0);

		GLES32.glEnable(GLES32.GL_DEPTH_TEST);
		GLES32.glDepthFunc(GLES32.GL_LEQUAL);

		GLES32.glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	}



	private void uninitialize(){


		if(vbo_Pyramid_Normal[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Pyramid_Normal, 0);
			vbo_Pyramid_Normal[0] = 0;
		}

		if(vbo_Pyramid_Position[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Pyramid_Position, 0);
			vbo_Pyramid_Position[0] = 0;
		}

		if(vao_Pyramid[0] != 0){
			GLES32.glDeleteVertexArrays(1, vao_Pyramid, 0);
			vao_Pyramid[0] = 0;
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




	private void resize(int width, int height){
		if(height == 0)
			height = 1;

		GLES32.glViewport(0, 0, width, height);

		Matrix.setIdentityM(perspectiveProjectionMatrix, 0);

		Matrix.perspectiveM(perspectiveProjectionMatrix, 0,
						45.0f,
						(float)width / (float)height,
						0.1f,
						100.0f);
	}


	private void display(){

		float translateMatrix[] = new float[4*4];
		float rotateMatrix[] = new float[4*4];
		float modelMatrix[] = new float[4*4];
		float viewMatrix[] = new float[4*4];


		GLES32.glClear(GLES32.GL_COLOR_BUFFER_BIT | GLES32.GL_DEPTH_BUFFER_BIT);

			

		if(iWhichLight == PER_VERTEX){
			
			GLES32.glUseProgram(shaderProgramObject_PV);
			
			Matrix.setIdentityM(translateMatrix, 0);
			Matrix.setIdentityM(rotateMatrix, 0);
			Matrix.setIdentityM(modelMatrix, 0);
			Matrix.setIdentityM(viewMatrix, 0);

			Matrix.translateM(translateMatrix, 0, 0.0f, 0.0f, -4.50f);
			Matrix.rotateM(rotateMatrix, 0, angle_pyramid, 0.0f, 1.0f, 0.0f);
			Matrix.multiplyMM(modelMatrix, 0, modelMatrix, 0, translateMatrix, 0);
			Matrix.multiplyMM(modelMatrix, 0, modelMatrix, 0, rotateMatrix, 0);

			GLES32.glUniformMatrix4fv(modelMatrixUniform_PV, 1, false, modelMatrix, 0);
			GLES32.glUniformMatrix4fv(viewMatrixUniform_PV, 1, false, viewMatrix, 0);
			GLES32.glUniformMatrix4fv(projectionMatrixUniform_PV, 1, false, perspectiveProjectionMatrix, 0);
			
			
			if(bLights == true){

					GLES32.glUniform1i(LKeyPressUniform_PV, 1);

					GLES32.glUniform3fv(red_la_Uniform_PV, 1, lightAmbient_Red, 0);
					GLES32.glUniform3fv(red_ld_Uniform_PV, 1, lightDiffuse_Red, 0);
					GLES32.glUniform3fv(red_ls_Uniform_PV, 1, lightSpecular_Red, 0);
					GLES32.glUniform4fv(red_lightPositionUniform_PV, 1, lightPosition_Red, 0);

					GLES32.glUniform3fv(blue_la_Uniform_PV, 1, lightAmbient_Blue, 0);
					GLES32.glUniform3fv(blue_ld_Uniform_PV, 1, lightDiffuse_Blue, 0);
					GLES32.glUniform3fv(blue_ls_Uniform_PV, 1, lightSpecular_Blue, 0);
					GLES32.glUniform4fv(blue_lightPositionUniform_PV, 1, lightPosition_Blue, 0);

					GLES32.glUniform3fv(ka_Uniform_PV, 1, materialAmbient, 0);
					GLES32.glUniform3fv(kd_Uniform_PV, 1, materialDiffuse, 0);
					GLES32.glUniform3fv(ks_Uniform_PV, 1, materialSpecular, 0);
					GLES32.glUniform1f(shininessUniform_PV, materialShininess);

			}
			else
				GLES32.glUniform1i(LKeyPressUniform_PV, 0);

			GLES32.glBindVertexArray(vao_Pyramid[0]);
				GLES32.glDrawArrays(GLES32.GL_TRIANGLES, 0, 12);
			GLES32.glBindVertexArray(0);

			GLES32.glUseProgram(0);
		}
		else{

			GLES32.glUseProgram(shaderProgramObject_PF);

			Matrix.setIdentityM(translateMatrix, 0);
			Matrix.setIdentityM(rotateMatrix, 0);
			Matrix.setIdentityM(modelMatrix, 0);
			Matrix.setIdentityM(viewMatrix, 0);

			Matrix.translateM(translateMatrix, 0, 0.0f, 0.0f, -4.50f);
			Matrix.rotateM(rotateMatrix, 0, angle_pyramid, 0.0f, 1.0f, 0.0f);
			Matrix.multiplyMM(modelMatrix, 0, modelMatrix, 0, translateMatrix, 0);
			Matrix.multiplyMM(modelMatrix, 0, modelMatrix, 0, rotateMatrix, 0);

			GLES32.glUniformMatrix4fv(modelMatrixUniform_PF, 1, false, modelMatrix, 0);
			GLES32.glUniformMatrix4fv(viewMatrixUniform_PF, 1, false, viewMatrix, 0);
			GLES32.glUniformMatrix4fv(projectionMatrixUniform_PF, 1, false, perspectiveProjectionMatrix, 0);
	
			if(bLights == true){

					GLES32.glUniform1i(LKeyPressUniform_PF, 1);

					GLES32.glUniform3fv(red_la_Uniform_PF, 1, lightAmbient_Red, 0);
					GLES32.glUniform3fv(red_ld_Uniform_PF, 1, lightDiffuse_Red, 0);
					GLES32.glUniform3fv(red_ls_Uniform_PF, 1, lightSpecular_Red, 0);
					GLES32.glUniform4fv(red_lightPositionUniform_PF, 1, lightPosition_Red, 0);

					GLES32.glUniform3fv(blue_la_Uniform_PF, 1, lightAmbient_Blue, 0);
					GLES32.glUniform3fv(blue_ld_Uniform_PF, 1, lightDiffuse_Blue, 0);
					GLES32.glUniform3fv(blue_ls_Uniform_PF, 1, lightSpecular_Blue, 0);
					GLES32.glUniform4fv(blue_lightPositionUniform_PF, 1, lightPosition_Blue, 0);

					GLES32.glUniform3fv(ka_Uniform_PF, 1, materialAmbient, 0);
					GLES32.glUniform3fv(kd_Uniform_PF, 1, materialDiffuse, 0);
					GLES32.glUniform3fv(ks_Uniform_PF, 1, materialSpecular, 0);
					GLES32.glUniform1f(shininessUniform_PF, materialShininess);

			}
			else
				GLES32.glUniform1i(LKeyPressUniform_PF, 0);

			GLES32.glBindVertexArray(vao_Pyramid[0]);
				GLES32.glDrawArrays(GLES32.GL_TRIANGLES, 0, 12);
			GLES32.glBindVertexArray(0);
			GLES32.glUseProgram(0);
		}



		requestRender();

	}


	private void update(){
		angle_pyramid = angle_pyramid + 1.0f;

		if(angle_pyramid > 360.0f)
			angle_pyramid = 0.0f;
	}
}