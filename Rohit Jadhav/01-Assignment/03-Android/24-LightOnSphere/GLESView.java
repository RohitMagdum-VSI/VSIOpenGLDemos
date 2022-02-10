package com.rohit_r_jadhav.light_on_sphere;

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

	GestureDetector gestureDetector;
	Context context;
	boolean bLights = false;

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
	private int vertexShaderObject;
	private int fragmentShaderObject;
	private int shaderProgramObject;

	//For Sphere
	private int vao_Sphere[] = new int[1];
	private int vbo_Sphere_Position[] = new int[1];
	private int vbo_Sphere_Normal[] = new int[1];
	private int vbo_Sphere_Element[] = new int[1];
	private float angle_sphere = 0.0f;
	int numVertices;
	int numElements;




	//For Projection
	float perspectiveProjectionMatrix[] = new float[4*4];

	//For Uniform
	private int modelMatrixUniform;
	private int viewMatrixUniform;
	private int projectionMatrixUniform;
	private int la_Uniform;
	private int ld_Uniform;
	private int ls_Uniform;
	private int lightPositionUniform;
	private int ka_Uniform;
	private int kd_Uniform;
	private int ks_Uniform;
	private int shininessUniform;
	private int LKeyPressUniform;

	//For Lights
	float lightAmbient[] = new float[]{0.0f, 0.0f, 0.0f, 0.0f};
	float lightDiffuse[] = new float[] {1.0f, 1.0f, 1.0f, 1.0f};
	float lightSpecular[] = new float[]{1.0f, 1.0f, 1.0f, 1.0f};
	float lightPosition[] = new float[]{100.0f, 100.0f, 100.0f, 1.0f};

	//For Material
	float materialAmbient[] = new float[]{0.0f, 0.0f, 0.0f, 0.0f};
	float materialDiffuse[] = new float[] {1.0f, 1.0f, 1.0f, 1.0f};
	float materialSpecular[] = new float[] {1.0f, 1.0f, 1.0f, 1.0f};
	float materialShininess = 128.0f;



	private void initialize(){


		vertexShaderObject = GLES32.glCreateShader(GLES32.GL_VERTEX_SHADER);

		final String vertexShaderCode = String.format(
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


		GLES32.glShaderSource(vertexShaderObject, vertexShaderCode);

		GLES32.glCompileShader(vertexShaderObject);

		int iShaderCompileStatus[] = new int[1];
		int iInfoLogLength[] = new int[1];
		String szInfoLog = null;

		GLES32.glGetShaderiv(vertexShaderObject, GLES32.GL_COMPILE_STATUS, iShaderCompileStatus, 0);
		if(iShaderCompileStatus[0] == GLES32.GL_FALSE){
			GLES32.glGetShaderiv(vertexShaderObject, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if(iInfoLogLength[0] > 0){
				szInfoLog = GLES32.glGetShaderInfoLog(vertexShaderObject);
				System.out.println("RTR: Vertex Shader Compilation Error: \n"+ szInfoLog);
				szInfoLog = null;
				uninitialize();
				System.exit(1);
			}
		}




		fragmentShaderObject = GLES32.glCreateShader(GLES32.GL_FRAGMENT_SHADER);
		
		final String fragmentShaderCode = String.format(
				"#version 320 es" +
				"\n" +
				"precision highp float;" +
				"in vec3 phongLight;" +
				"out vec4 FragColor;" +
				"void main(void)" +
				"{" +
					"FragColor = vec4(phongLight, 1.0f);" +
				"}"
			);

		GLES32.glShaderSource(fragmentShaderObject, fragmentShaderCode);
		GLES32.glCompileShader(fragmentShaderObject);

		iShaderCompileStatus[0] = 0;
		iInfoLogLength[0] = 0;
		szInfoLog = null;

		GLES32.glGetShaderiv(fragmentShaderObject, GLES32.GL_COMPILE_STATUS, iShaderCompileStatus, 0);
		if(iShaderCompileStatus[0] == GLES32.GL_FALSE){
			GLES32.glGetShaderiv(fragmentShaderObject, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if(iInfoLogLength[0] > 0){
				szInfoLog = GLES32.glGetShaderInfoLog(fragmentShaderObject);
				System.out.println("RTR: Fragment Shader Compilation Error: \n"+ szInfoLog);
				szInfoLog = null;
				uninitialize();
				System.exit(1);
			}
		}



		shaderProgramObject = GLES32.glCreateProgram();

		GLES32.glAttachShader(shaderProgramObject, vertexShaderObject);
		GLES32.glAttachShader(shaderProgramObject, fragmentShaderObject);

		GLES32.glBindAttribLocation(shaderProgramObject, GLESMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
		GLES32.glBindAttribLocation(shaderProgramObject, GLESMacros.AMC_ATTRIBUTE_NORMAL, "vNormal");

		GLES32.glLinkProgram(shaderProgramObject);

		int iProgramLinkStatus[] = new int[1];
		iInfoLogLength[0] = 0;
		szInfoLog = null;

		GLES32.glGetProgramiv(shaderProgramObject, GLES32.GL_LINK_STATUS, iProgramLinkStatus, 0);
		if(iProgramLinkStatus[0] == GLES32.GL_FALSE){
			GLES32.glGetProgramiv(shaderProgramObject, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if(iInfoLogLength[0] > 0){
				szInfoLog = GLES32.glGetProgramInfoLog(shaderProgramObject);
				System.out.println("RTR: Shader Program Linking Error: \n" + szInfoLog);
				szInfoLog = null;
				uninitialize();
				System.exit(1);
			} 
		}



		modelMatrixUniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_model_matrix");
		viewMatrixUniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_view_matrix");
		projectionMatrixUniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_projection_matrix");
		la_Uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_la");
		ld_Uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_ld");
		ls_Uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_ls");
		lightPositionUniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_light_position");

		ka_Uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_ka");
		kd_Uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_kd");
		ks_Uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_ks");
		shininessUniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_shininess");
		LKeyPressUniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_LKeyPress");


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

		GLES32.glClearColor(0.0f, 0.0f, 0.05f, 0.0f);

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


		if(shaderProgramObject != 0){

			GLES32.glUseProgram(shaderProgramObject);

				/*int iShaderCount[] = new int[1];
				int iShaderNo = 0;
				GLES32.glGetProgramiv(shaderProgramObject, GLES32.GL_ATTACHED_SHADERS, iShaderCount, 0);
				System.out.println("RTR: ShaderCount: " + iShaderCount[0]);
				int iShaders[] = new int[iShaderCount[0]];
				GLES32.glGetAttachedShaders(shaderProgramObject, iShaderCount[0], iShaderCount, 0, iShaders, 0);
				for(iShaderNo =0; iShaderNo < iShaderCount[0]; iShaderNo++){
					GLES32.glDetachShader(shaderProgramObject, iShaders[iShaderNo]);
					GLES32.glDeleteShader(iShaders[iShaderNo]);
					iShaders[iShaderNo] = 0;
				}*/

				if(fragmentShaderObject != 0){
					GLES32.glDetachShader(shaderProgramObject, fragmentShaderObject);
					GLES32.glDeleteShader(fragmentShaderObject);
					fragmentShaderObject= 0;
				}

				if(vertexShaderObject != 0){
					GLES32.glDetachShader(shaderProgramObject, vertexShaderObject);
					GLES32.glDeleteShader(vertexShaderObject);
					vertexShaderObject = 0;
				}

			GLES32.glUseProgram(0);
			GLES32.glDeleteProgram(shaderProgramObject);
			shaderProgramObject = 0; 
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


		GLES32.glUseProgram(shaderProgramObject);

			Matrix.setIdentityM(translateMatrix, 0);
			Matrix.setIdentityM(rotateMatrix, 0);
			Matrix.setIdentityM(modelMatrix, 0);
			Matrix.setIdentityM(viewMatrix, 0);

			Matrix.translateM(translateMatrix, 0, 0.0f, 0.0f, -1.50f);
			Matrix.rotateM(rotateMatrix, 0, angle_sphere, 0.0f, 1.0f, 0.0f);
			Matrix.multiplyMM(modelMatrix, 0, modelMatrix, 0, translateMatrix, 0);
			Matrix.multiplyMM(modelMatrix, 0, modelMatrix, 0, rotateMatrix, 0);

			GLES32.glUniformMatrix4fv(modelMatrixUniform, 1, false, modelMatrix, 0);
			GLES32.glUniformMatrix4fv(viewMatrixUniform, 1, false, viewMatrix, 0);
			GLES32.glUniformMatrix4fv(projectionMatrixUniform, 1, false, perspectiveProjectionMatrix, 0);

			if(bLights == true){

					GLES32.glUniform1i(LKeyPressUniform, 1);

					GLES32.glUniform3fv(la_Uniform, 1, lightAmbient, 0);
					GLES32.glUniform3fv(ld_Uniform, 1, lightDiffuse, 0);
					GLES32.glUniform3fv(ls_Uniform, 1, lightSpecular, 0);
					GLES32.glUniform4fv(lightPositionUniform, 1, lightPosition, 0);

					GLES32.glUniform3fv(ka_Uniform, 1, materialAmbient, 0);
					GLES32.glUniform3fv(kd_Uniform, 1, materialDiffuse, 0);
					GLES32.glUniform3fv(ks_Uniform, 1, materialSpecular, 0);
					GLES32.glUniform1f(shininessUniform, materialShininess);

			}
			else
				GLES32.glUniform1i(LKeyPressUniform, 0);


		GLES32.glBindVertexArray(vao_Sphere[0]);

			GLES32.glBindBuffer(GLES32.GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element[0]);
			GLES32.glDrawElements(GLES32.GL_TRIANGLES, numElements, GLES32.GL_UNSIGNED_SHORT, 0);

		GLES32.glBindVertexArray(0);

		GLES32.glUseProgram(0);

		requestRender();

	}

	private void update(){
		angle_sphere = angle_sphere + .50f;
		if(angle_sphere > 360.0f)
			angle_sphere = 0.0f;
	}

}
