package com.rohit_r_jadhav.spot_light;

import android.opengl.GLSurfaceView;
import android.opengl.GLES32;
import android.opengl.Matrix;

import java.lang.Math;

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

	GestureDetector gestureDetector_RRJ;
	Context context_RRJ;
	boolean bLights = false;

	GLESView(Context drawingContext_RRJ){
		super(drawingContext_RRJ);
		context_RRJ = drawingContext_RRJ;

		gestureDetector_RRJ = new GestureDetector(drawingContext_RRJ, this, null, false);
		gestureDetector_RRJ.setOnDoubleTapListener(this);

		setEGLContextClientVersion(3);
		setRenderer(this);
		setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
	}

	@Override
	public boolean onTouchEvent(MotionEvent e){
		int action = e.getAction();
		if(!gestureDetector_RRJ.onTouchEvent(e))
			super.onTouchEvent(e);
		return(true);
	}

	

	/********** Methods from OnDoubleTapListener **********/
	@Override
	public boolean onDoubleTap(MotionEvent e){
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
	private int vertexShaderObject_RRJ;
	private int fragmentShaderObject_RRJ;
	private int shaderProgramObject_RRJ;

	//For Sphere
	private int vao_Sphere_RRJ[] = new int[1];
	private int vbo_Sphere_Position_RRJ[] = new int[1];
	private int vbo_Sphere_Normal_RRJ[] = new int[1];
	private int vbo_Sphere_Element_RRJ[] = new int[1];
	private float angle_sphere_RRJ = 0.0f;
	private int numVertices_RRJ;
	private int numElements_RRJ;




	//For Projection
	private float perspectiveProjectionMatrix_RRJ[] = new float[4*4];

	//For Uniform
	private int modelMatrixUniform_RRJ;
	private int viewMatrixUniform_RRJ;
	private int projectionMatrixUniform_RRJ;
	private int la_Uniform_RRJ;
	private int ld_Uniform_RRJ;
	private int ls_Uniform_RRJ;
	private int lightPositionUniform_RRJ;
	private int ka_Uniform_RRJ;
	private int kd_Uniform_RRJ;
	private int ks_Uniform_RRJ;
	private int shininessUniform_RRJ;
	private int LKeyPressUniform_RRJ;

	//For Lights
	private float lightAmbient_RRJ[] = new float[]{0.0f, 0.0f, 0.0f, 0.0f};
	private float lightDiffuse_RRJ[] = new float[] {1.0f, 1.0f, 1.0f, 1.0f};
	private float lightSpecular_RRJ[] = new float[]{1.0f, 1.0f, 1.0f, 1.0f};
	private float lightPosition_RRJ[] = new float[]{0.0f, 0.0f, 5.0f, 1.0f};

	//For Material
	private float materialAmbient_RRJ[] = new float[]{0.0f, 0.0f, 0.0f, 0.0f};
	private float materialDiffuse_RRJ[] = new float[] {1.0f, 1.0f, 1.0f, 1.0f};
	private float materialSpecular_RRJ[] = new float[] {1.0f, 1.0f, 1.0f, 1.0f};
	private float materialShininess_RRJ = 50.0f;



	//For Spot Light Uniform
	private int spotLightDirectionUniform_RRJ;
	private int spotLightCutoffUniform_RRJ;
	private int spotLightExponentUniform_RRJ;
	private int constantAttenuationUniform_RRJ;
	private int linearAttenuationUniform_RRJ;
	private int quadraticAttenuationUniform_RRJ;

	//For Spot Light Values
	final float PI = 3.1415926535f;
	private float spotLightDirection_RRJ[] = new float[]{0.0f, 0.0f, -1.0f, 1.0f};
	private float angle_RRJ = (float)(2.0f * Math.PI / 180.0f);
	private float spotLightCutoff_RRJ = (float)Math.cos(angle_RRJ);	//0.78539f
	private float spotLightExponent_RRJ = 20.0f;
	private float constantAttenuation_RRJ = 1.0f;
	private float linearAttenuation_RRJ = 0.09f;
	private float quadraticAttenuation_RRJ = 0.032f;



	private void initialize(){


		vertexShaderObject_RRJ = GLES32.glCreateShader(GLES32.GL_VERTEX_SHADER);

		final String vertexShaderCode_RRJ = String.format(
				"#version 320 es" +
				"\n" +
				"in vec4 vPosition;" +
				"in vec3 vNormal;" +
				"uniform mat4 u_model_matrix;" +
				"uniform mat4 u_view_matrix;" +
				"uniform mat4 u_projection_matrix;" +
				"uniform vec4 u_light_position;" +

				"out vec3 lightDirection_VS;" +
				"out vec3 viewerVec_VS;" +
				"out vec3 transformedNormal_VS;" +
				
				"void main(void)" +
				"{" +
					
					"vec4 eyeCoord = u_view_matrix * u_model_matrix * vPosition;" +

					"lightDirection_VS = vec3(u_light_position - eyeCoord);" +
					"mat3 normalMatrix = mat3(u_view_matrix * u_model_matrix);" +
					"transformedNormal_VS = (normalMatrix * vNormal);" +
					"viewerVec_VS = (-eyeCoord.xyz);" +

					"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" +

				"}"  
			);


		GLES32.glShaderSource(vertexShaderObject_RRJ, vertexShaderCode_RRJ);

		GLES32.glCompileShader(vertexShaderObject_RRJ);

		int iShaderCompileStatus_RRJ[] = new int[1];
		int iInfoLogLength_RRJ[] = new int[1];
		String szInfoLog_RRJ = null;

		GLES32.glGetShaderiv(vertexShaderObject_RRJ, GLES32.GL_COMPILE_STATUS, iShaderCompileStatus_RRJ, 0);
		if(iShaderCompileStatus_RRJ[0] == GLES32.GL_FALSE){
			GLES32.glGetShaderiv(vertexShaderObject_RRJ, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength_RRJ, 0);
			if(iInfoLogLength_RRJ[0] > 0){
				szInfoLog_RRJ = GLES32.glGetShaderInfoLog(vertexShaderObject_RRJ);
				System.out.println("RTR: Vertex Shader Compilation Error:"+ szInfoLog_RRJ);
				szInfoLog_RRJ = null;
				uninitialize();
				System.exit(1);
			}
		}




		fragmentShaderObject_RRJ = GLES32.glCreateShader(GLES32.GL_FRAGMENT_SHADER);
		
		final String fragmentShaderCode_RRJ = String.format(
				"#version 320 es" +
				"\n" +
				"precision highp float;" +

				"in vec3 lightDirection_VS;" +
				"in vec3 transformedNormal_VS;" +
				"in vec3 viewerVec_VS;" +


				"uniform vec4 u_spotLightDirection;" +
				"uniform float u_spotLightCutoff;" +
				"uniform float u_spotLightExponent;" +
				"uniform float u_constantAttenuation;" +
				"uniform float u_linearAttenuation;" +
				"uniform float u_quadraticAttenuation;" +


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

					"if(u_LKeyPress == 1) {" +

						"vec3 normalize_LightDirection = normalize(lightDirection_VS);" +
						"vec3 normalize_TransformedNormal = normalize(transformedNormal_VS);" +
						"float S_Dot_N = max(dot(normalize_LightDirection, normalize_TransformedNormal), 0.0);" +

						"vec3 normalize_ViewerVec = normalize(viewerVec_VS);" +
						"vec3 reflectionVec = reflect(-normalize_LightDirection, normalize_TransformedNormal);" +
						"float R_Dot_V = max(dot(reflectionVec, normalize_ViewerVec), 0.0);" +

						"float d = length(normalize_LightDirection);" +
						"float attenuation = 1.0f / (u_quadraticAttenuation * d * d + u_linearAttenuation * d + u_constantAttenuation);" +

						"float spotDot = max(dot(-normalize_LightDirection, normalize(u_spotLightDirection.xyz)), 0.0);" +
						"float attenuationFactor;" +

						"if(spotDot > u_spotLightCutoff) {" +
							"attenuationFactor = pow(spotDot, u_spotLightExponent);" +
						"}" +
						"else {" +
							"attenuationFactor = 0.5f;" +
						"}" +

						"attenuation = attenuationFactor * attenuation;" +

						"vec3 ambient = u_la * u_ka * attenuation;" +
						"vec3 diffuse = u_ld * u_kd * S_Dot_N * attenuation;" +
						"vec3 specular = u_ls * u_ks * pow(R_Dot_V, u_shininess) * attenuation;" +

						"vec3 phong_ADS_light = ambient + diffuse + specular;" +

						"FragColor = vec4(phong_ADS_light, 1.0);" +

					"}" +
					"else {" +
						"FragColor = vec4(1.0, 1.0, 1.0, 1.0);" +
					"}" +
				"}"
			);

		GLES32.glShaderSource(fragmentShaderObject_RRJ, fragmentShaderCode_RRJ);
		GLES32.glCompileShader(fragmentShaderObject_RRJ);

		iShaderCompileStatus_RRJ[0] = 0;
		iInfoLogLength_RRJ[0] = 0;
		szInfoLog_RRJ = null;

		GLES32.glGetShaderiv(fragmentShaderObject_RRJ, GLES32.GL_COMPILE_STATUS, iShaderCompileStatus_RRJ, 0);
		if(iShaderCompileStatus_RRJ[0] == GLES32.GL_FALSE){
			GLES32.glGetShaderiv(fragmentShaderObject_RRJ, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength_RRJ, 0);
			if(iInfoLogLength_RRJ[0] > 0){
				szInfoLog_RRJ = GLES32.glGetShaderInfoLog(fragmentShaderObject_RRJ);
				System.out.println("RTR: Fragment Shader Compilation Error:"+ szInfoLog_RRJ);
				szInfoLog_RRJ = null;
				uninitialize();
				System.exit(1);
			}
		}



		shaderProgramObject_RRJ = GLES32.glCreateProgram();

		GLES32.glAttachShader(shaderProgramObject_RRJ, vertexShaderObject_RRJ);
		GLES32.glAttachShader(shaderProgramObject_RRJ, fragmentShaderObject_RRJ);

		GLES32.glBindAttribLocation(shaderProgramObject_RRJ, GLESMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
		GLES32.glBindAttribLocation(shaderProgramObject_RRJ, GLESMacros.AMC_ATTRIBUTE_NORMAL, "vNormal");

		GLES32.glLinkProgram(shaderProgramObject_RRJ);

		int iProgramLinkStatus_RRJ[] = new int[1];
		iInfoLogLength_RRJ[0] = 0;
		szInfoLog_RRJ = null;

		GLES32.glGetProgramiv(shaderProgramObject_RRJ, GLES32.GL_LINK_STATUS, iProgramLinkStatus_RRJ, 0);
		if(iProgramLinkStatus_RRJ[0] == GLES32.GL_FALSE){
			GLES32.glGetProgramiv(shaderProgramObject_RRJ, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength_RRJ, 0);
			if(iInfoLogLength_RRJ[0] > 0){
				szInfoLog_RRJ = GLES32.glGetProgramInfoLog(shaderProgramObject_RRJ);
				System.out.println("RTR: Shader Program Linking Error:" + szInfoLog_RRJ);
				szInfoLog_RRJ = null;
				uninitialize();
				System.exit(1);
			} 
		}



		modelMatrixUniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_model_matrix");
		viewMatrixUniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_view_matrix");
		projectionMatrixUniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_projection_matrix");
		la_Uniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_la");
		ld_Uniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_ld");
		ls_Uniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_ls");
		lightPositionUniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_light_position");

		ka_Uniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_ka");
		kd_Uniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_kd");
		ks_Uniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_ks");
		shininessUniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_shininess");
		LKeyPressUniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_LKeyPress");


		spotLightDirectionUniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_spotLightDirection");
		spotLightCutoffUniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_spotLightCutoff");
		spotLightExponentUniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_spotLightExponent");
		constantAttenuationUniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_constantAttenuation");
		linearAttenuationUniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_linearAttenuation");
		quadraticAttenuationUniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_quadraticAttenuation");


		Sphere sphere_RRJ = new Sphere();
		float sphere_Position_RRJ[] = new float[1146];
		float sphere_Normal_RRJ[] = new float[1146];
		float sphere_TexCoord_RRJ[] = new float[764];
		short sphere_Element_RRJ[] = new short[2280];
		


		sphere_RRJ.getSphereVertexData(sphere_Position_RRJ, sphere_Normal_RRJ, sphere_TexCoord_RRJ, sphere_Element_RRJ);
		numVertices_RRJ = sphere_RRJ.getNumberOfSphereVertices();
		numElements_RRJ = sphere_RRJ.getNumberOfSphereElements();


		GLES32.glGenVertexArrays(1, vao_Sphere_RRJ, 0);
		GLES32.glBindVertexArray(vao_Sphere_RRJ[0]);

			/********** Position **********/
			GLES32.glGenBuffers(1, vbo_Sphere_Position_RRJ, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Sphere_Position_RRJ[0]);

				ByteBuffer spherePosition_ByteBuffer = ByteBuffer.allocateDirect(sphere_Position_RRJ.length * 4);
				spherePosition_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer spherePosition_FloatBuffer = spherePosition_ByteBuffer.asFloatBuffer();
				spherePosition_FloatBuffer.put(sphere_Position_RRJ);
				spherePosition_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							sphere_Position_RRJ.length * 4,
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
			GLES32.glGenBuffers(1, vbo_Sphere_Normal_RRJ, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Sphere_Normal_RRJ[0]);

				ByteBuffer sphereNormal_ByteBuffer = ByteBuffer.allocateDirect(sphere_Normal_RRJ.length * 4);
				sphereNormal_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer sphereNormal_FloatBuffer = sphereNormal_ByteBuffer.asFloatBuffer();
				sphereNormal_FloatBuffer.put(sphere_Normal_RRJ);
				sphereNormal_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							sphere_Normal_RRJ.length * 4,
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
			GLES32.glGenBuffers(1, vbo_Sphere_Element_RRJ, 0);
			GLES32.glBindBuffer(GLES32.GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element_RRJ[0]);

				ByteBuffer sphereElement_ByteBuffer = ByteBuffer.allocateDirect(sphere_Element_RRJ.length * 4);
				sphereElement_ByteBuffer.order(ByteOrder.nativeOrder());
				ShortBuffer sphereElement_ShortBuffer = sphereElement_ByteBuffer.asShortBuffer();
				sphereElement_ShortBuffer.put(sphere_Element_RRJ);
				sphereElement_ShortBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ELEMENT_ARRAY_BUFFER,
							sphere_Element_RRJ.length * 4,
							sphereElement_ShortBuffer,
							GLES32.GL_STATIC_DRAW);
			GLES32.glBindBuffer(GLES32.GL_ELEMENT_ARRAY_BUFFER, 0);


		GLES32.glBindVertexArray(0);

		GLES32.glEnable(GLES32.GL_DEPTH_TEST);
		GLES32.glDepthFunc(GLES32.GL_LEQUAL);

		GLES32.glClearColor(0.0f, 0.0f, 0.05f, 0.0f);

	}

	private void uninitialize(){


		if(vbo_Sphere_Element_RRJ[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Sphere_Element_RRJ, 0);
			vbo_Sphere_Element_RRJ[0] = 0;
		}

		if(vbo_Sphere_Normal_RRJ[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Sphere_Normal_RRJ, 0);
			vbo_Sphere_Normal_RRJ[0] = 0;
		}

		if(vbo_Sphere_Position_RRJ[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Sphere_Position_RRJ, 0);
			vbo_Sphere_Position_RRJ[0] = 0;
		}

		if(vao_Sphere_RRJ[0] != 0){
			GLES32.glDeleteVertexArrays(1, vao_Sphere_RRJ, 0);
			vao_Sphere_RRJ[0] = 0;
		}


		if(shaderProgramObject_RRJ != 0){

			GLES32.glUseProgram(shaderProgramObject_RRJ);

				/*int iShaderCount[] = new int[1];
				int iShaderNo = 0;
				GLES32.glGetProgramiv(shaderProgramObject_RRJ, GLES32.GL_ATTACHED_SHADERS, iShaderCount, 0);
				System.out.println("RTR: ShaderCount: " + iShaderCount[0]);
				int iShaders[] = new int[iShaderCount[0]];
				GLES32.glGetAttachedShaders(shaderProgramObject_RRJ, iShaderCount[0], iShaderCount, 0, iShaders, 0);
				for(iShaderNo =0; iShaderNo < iShaderCount[0]; iShaderNo++){
					GLES32.glDetachShader(shaderProgramObject_RRJ, iShaders[iShaderNo]);
					GLES32.glDeleteShader(iShaders[iShaderNo]);
					iShaders[iShaderNo] = 0;
				}*/

				if(fragmentShaderObject_RRJ != 0){
					GLES32.glDetachShader(shaderProgramObject_RRJ, fragmentShaderObject_RRJ);
					GLES32.glDeleteShader(fragmentShaderObject_RRJ);
					fragmentShaderObject_RRJ= 0;
				}

				if(vertexShaderObject_RRJ != 0){
					GLES32.glDetachShader(shaderProgramObject_RRJ, vertexShaderObject_RRJ);
					GLES32.glDeleteShader(vertexShaderObject_RRJ);
					vertexShaderObject_RRJ = 0;
				}

			GLES32.glUseProgram(0);
			GLES32.glDeleteProgram(shaderProgramObject_RRJ);
			shaderProgramObject_RRJ = 0; 
		}
	}


	private void resize(int width, int height){
		if(height == 0)
			height = 1;

		GLES32.glViewport(0, 0, width, height);

		Matrix.setIdentityM(perspectiveProjectionMatrix_RRJ, 0);

		Matrix.perspectiveM(perspectiveProjectionMatrix_RRJ, 0,
						45.0f,
						(float)width / (float)height,
						0.1f,
						100.0f);
	}



	private void display(){

		float translateMatrix_RRJ[] = new float[4*4];
		float rotateMatrix_RRJ[] = new float[4*4];
		float modelMatrix_RRJ[] = new float[4*4];
		float viewMatrix_RRJ[] = new float[4*4];


		GLES32.glClear(GLES32.GL_COLOR_BUFFER_BIT | GLES32.GL_DEPTH_BUFFER_BIT);


		GLES32.glUseProgram(shaderProgramObject_RRJ);

			Matrix.setIdentityM(translateMatrix_RRJ, 0);
			Matrix.setIdentityM(rotateMatrix_RRJ, 0);
			Matrix.setIdentityM(modelMatrix_RRJ, 0);
			Matrix.setIdentityM(viewMatrix_RRJ, 0);

			Matrix.translateM(translateMatrix_RRJ, 0, 0.0f, 0.0f, -1.50f);
			Matrix.rotateM(rotateMatrix_RRJ, 0, angle_sphere_RRJ, 0.0f, 1.0f, 0.0f);
			Matrix.multiplyMM(modelMatrix_RRJ, 0, modelMatrix_RRJ, 0, translateMatrix_RRJ, 0);
			Matrix.multiplyMM(modelMatrix_RRJ, 0, modelMatrix_RRJ, 0, rotateMatrix_RRJ, 0);

			GLES32.glUniformMatrix4fv(modelMatrixUniform_RRJ, 1, false, modelMatrix_RRJ, 0);
			GLES32.glUniformMatrix4fv(viewMatrixUniform_RRJ, 1, false, viewMatrix_RRJ, 0);
			GLES32.glUniformMatrix4fv(projectionMatrixUniform_RRJ, 1, false, perspectiveProjectionMatrix_RRJ, 0);

			if(bLights == true){

					GLES32.glUniform1i(LKeyPressUniform_RRJ, 1);

					GLES32.glUniform3fv(la_Uniform_RRJ, 1, lightAmbient_RRJ, 0);
					GLES32.glUniform3fv(ld_Uniform_RRJ, 1, lightDiffuse_RRJ, 0);
					GLES32.glUniform3fv(ls_Uniform_RRJ, 1, lightSpecular_RRJ, 0);
					GLES32.glUniform4fv(lightPositionUniform_RRJ, 1, lightPosition_RRJ, 0);

					GLES32.glUniform3fv(ka_Uniform_RRJ, 1, materialAmbient_RRJ, 0);
					GLES32.glUniform3fv(kd_Uniform_RRJ, 1, materialDiffuse_RRJ, 0);
					GLES32.glUniform3fv(ks_Uniform_RRJ, 1, materialSpecular_RRJ, 0);
					GLES32.glUniform1f(shininessUniform_RRJ, materialShininess_RRJ);


					GLES32.glUniform4fv(spotLightDirectionUniform_RRJ, 1, spotLightDirection_RRJ, 0);
					GLES32.glUniform1f(spotLightCutoffUniform_RRJ, spotLightCutoff_RRJ);
					GLES32.glUniform1f(spotLightExponentUniform_RRJ, spotLightExponent_RRJ);

					GLES32.glUniform1f(constantAttenuationUniform_RRJ, constantAttenuation_RRJ);
					GLES32.glUniform1f(linearAttenuationUniform_RRJ, linearAttenuation_RRJ);
					GLES32.glUniform1f(quadraticAttenuationUniform_RRJ, quadraticAttenuation_RRJ);

			}
			else
				GLES32.glUniform1i(LKeyPressUniform_RRJ, 0);


		GLES32.glBindVertexArray(vao_Sphere_RRJ[0]);

			GLES32.glBindBuffer(GLES32.GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element_RRJ[0]);
			GLES32.glDrawElements(GLES32.GL_TRIANGLES, numElements_RRJ, GLES32.GL_UNSIGNED_SHORT, 0);

		GLES32.glBindVertexArray(0);

		GLES32.glUseProgram(0);

		requestRender();

	}

	private void update(){
		angle_sphere_RRJ = angle_sphere_RRJ + .50f;
		if(angle_sphere_RRJ > 360.0f)
			angle_sphere_RRJ = 0.0f;
	}

}
