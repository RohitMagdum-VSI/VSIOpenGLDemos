package com.rohit_r_jadhav.solar_system_withmoon;

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


import java.io.*;
import java.util.*;

public class GLESView extends GLSurfaceView implements GLSurfaceView.Renderer, OnDoubleTapListener, OnGestureListener{

	GestureDetector gestureDetector_RRJ;
	Context context_RRJ;

	final int YEAR_RRJ = 1;
	final int DAY_RRJ = 2;
	final int MOON_RRJ = 3;

	final int CLKWISE_RRJ = 3;
	final int ANTICLKWISE_RRJ = 4;

	int iWhichRotation_RRJ = YEAR_RRJ;
	int iWhichDirection_RRJ = CLKWISE_RRJ;

	int year_RRJ = 0;
	int day_RRJ = 0;
	int moon_RRJ = 0;

	GLESView(Context drawingContext){
		super(drawingContext);
		context_RRJ = drawingContext;

		gestureDetector_RRJ = new GestureDetector(drawingContext, this, null, false);
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

		if(iWhichDirection_RRJ == CLKWISE_RRJ)
			iWhichDirection_RRJ = ANTICLKWISE_RRJ;
		else
			iWhichDirection_RRJ = CLKWISE_RRJ;

		return(true);
	}

	@Override
	public boolean onDoubleTapEvent(MotionEvent e){
		return(true);
	}

	@Override
	public boolean onSingleTapConfirmed(MotionEvent e){



		if(iWhichRotation_RRJ == YEAR_RRJ)
			iWhichRotation_RRJ = DAY_RRJ;
		else if(iWhichRotation_RRJ == DAY_RRJ)
			iWhichRotation_RRJ = MOON_RRJ;
		else 
			iWhichRotation_RRJ = YEAR_RRJ;



		/*if(iWhichRotation_RRJ == YEAR_RRJ){
			if(iWhichDirection_RRJ == CLKWISE_RRJ)
				year_RRJ = (year_RRJ + 5) % 360;
			else
				year_RRJ = (year_RRJ - 5) % 360;
		}
		else if(iWhichRotation_RRJ == DAY_RRJ){
			if(iWhichDirection_RRJ == ANTICLKWISE_RRJ)
				day_RRJ = (day_RRJ + 10) % 360;
			else
				day_RRJ = (day_RRJ + 10) % 360;
			
		}*/

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

		String version_RRJ = gl.glGetString(GL10.GL_VERSION);
		String glsl_version_RRJ = gl.glGetString(GLES32.GL_SHADING_LANGUAGE_VERSION);
		String renderer_RRJ = gl.glGetString(GL10.GL_RENDERER);
		String vendor_RRJ = gl.glGetString(GL10.GL_VENDOR);

		System.out.println("RTR: OpenGL Version: " + version_RRJ);
		System.out.println("RTR: OpenGLSL Verion: " + glsl_version_RRJ);
		System.out.println("RTR: Renderer: " + renderer_RRJ);
		System.out.println("RTR: Vendor: "+ vendor_RRJ);

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
	private int vbo_Sphere_Color_RRJ[] = new int[1];
	private int vbo_Sphere_Element_RRJ[] = new int[1];
	private float angle_sphere_RRJ = 0.0f;
	int numVertices_RRJ;
	int numElements_RRJ;


	//For Matrix Stack
	private float my_ModelViewMatrixStack_RRJ[] = new float[32 * 4 *4];
	private int iStackTop_RRJ = 0;



	//For Projection
	float perspectiveProjectionMatrix_RRJ[] = new float[4*4];

	//For Uniform
	private int projectionMatrixUniform_RRJ;
	private int modelViewUniform_RRJ;




	private void initialize(){


		vertexShaderObject_RRJ = GLES32.glCreateShader(GLES32.GL_VERTEX_SHADER);

		final String vertexShaderCode_RRJ = String.format(
				"#version  320 es" +
				"\n" +
				"in vec4 vPosition;" +
				"in vec3 vColor;" +
				"out vec3 outColor;" +
				
				"uniform mat4 u_modelview_matrix;" +
				"uniform mat4 u_projection_matrix;" +
				"void main(void) " +
				"{" +
					"outColor = vColor;" +
					"gl_Position = u_projection_matrix * u_modelview_matrix * vPosition;" +
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
				System.out.println("RTR: Vertex Shader Compilation Error: "+ szInfoLog_RRJ);
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
				"in vec3 outColor;" +
				"out vec4 FragColor;" +
				"void main(void)" +
				"{" +
					"FragColor = vec4(outColor, 1.0);" +
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
				System.out.println("RTR: Fragment Shader Compilation Error: "+ szInfoLog_RRJ);
				szInfoLog_RRJ = null;
				uninitialize();
				System.exit(1);
			}
		}



		shaderProgramObject_RRJ = GLES32.glCreateProgram();

		GLES32.glAttachShader(shaderProgramObject_RRJ, vertexShaderObject_RRJ);
		GLES32.glAttachShader(shaderProgramObject_RRJ, fragmentShaderObject_RRJ);

		GLES32.glBindAttribLocation(shaderProgramObject_RRJ, GLESMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
		GLES32.glBindAttribLocation(shaderProgramObject_RRJ, GLESMacros.AMC_ATTRIBUTE_COLOR, "vColor");

		GLES32.glLinkProgram(shaderProgramObject_RRJ);

		int iProgramLinkStatus_RRJ[] = new int[1];
		iInfoLogLength_RRJ[0] = 0;
		szInfoLog_RRJ = null;

		GLES32.glGetProgramiv(shaderProgramObject_RRJ, GLES32.GL_LINK_STATUS, iProgramLinkStatus_RRJ, 0);
		if(iProgramLinkStatus_RRJ[0] == GLES32.GL_FALSE){
			GLES32.glGetProgramiv(shaderProgramObject_RRJ, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength_RRJ, 0);
			if(iInfoLogLength_RRJ[0] > 0){
				szInfoLog_RRJ = GLES32.glGetProgramInfoLog(shaderProgramObject_RRJ);
				System.out.println("RTR: Shader Program Linking Error: " + szInfoLog_RRJ);
				szInfoLog_RRJ = null;
				uninitialize();
				System.exit(1);
			} 
		}



		
		modelViewUniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_modelview_matrix");
		projectionMatrixUniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_projection_matrix");
		


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

				ByteBuffer spherePosition_ByteBuffer_RRJ = ByteBuffer.allocateDirect(sphere_Position_RRJ.length * 4);
				spherePosition_ByteBuffer_RRJ.order(ByteOrder.nativeOrder());
				FloatBuffer spherePosition_FloatBuffer_RRJ = spherePosition_ByteBuffer_RRJ.asFloatBuffer();
				spherePosition_FloatBuffer_RRJ.put(sphere_Position_RRJ);
				spherePosition_FloatBuffer_RRJ.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							sphere_Position_RRJ.length * 4,
							spherePosition_FloatBuffer_RRJ,
							GLES32.GL_STATIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);



			/********** Color **********/
			GLES32.glVertexAttrib3f(GLESMacros.AMC_ATTRIBUTE_COLOR, 0.0f, 0.0f, 1.0f);
			/*GLES32.glGenBuffers(1, vbo_Sphere_Color_RRJ, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Sphere_Color_RRJ[0]);

				ByteBuffer sphereColor_ByteBuffer = ByteBuffer.allocateDirect(sphere_Color.length * 4);
				sphereColor_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer sphereColor_FloatBuffer = sphereColor_ByteBuffer.asFloatBuffer();
				sphereColor_FloatBuffer.put(sphere_Color);
				sphereColor_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							1146 * 4,
							null,
							GLES32.GL_DYNAMIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_COLOR,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_COLOR);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);*/


			/********** Elements **********/
			GLES32.glGenBuffers(1, vbo_Sphere_Element_RRJ, 0);
			GLES32.glBindBuffer(GLES32.GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element_RRJ[0]);

				ByteBuffer sphereElement_ByteBuffer_RRJ = ByteBuffer.allocateDirect(sphere_Element_RRJ.length * 4);
				sphereElement_ByteBuffer_RRJ.order(ByteOrder.nativeOrder());
				ShortBuffer sphereElement_ShortBuffer_RRJ = sphereElement_ByteBuffer_RRJ.asShortBuffer();
				sphereElement_ShortBuffer_RRJ.put(sphere_Element_RRJ);
				sphereElement_ShortBuffer_RRJ.position(0);

			GLES32.glBufferData(GLES32.GL_ELEMENT_ARRAY_BUFFER,
							sphere_Element_RRJ.length * 4,
							sphereElement_ShortBuffer_RRJ,
							GLES32.GL_STATIC_DRAW);
			GLES32.glBindBuffer(GLES32.GL_ELEMENT_ARRAY_BUFFER, 0);


		GLES32.glBindVertexArray(0);

		perspectiveProjectionMatrix_RRJ[0] = 454.0f;




		GLES32.glEnable(GLES32.GL_DEPTH_TEST);
		GLES32.glDepthFunc(GLES32.GL_LEQUAL);

		GLES32.glClearColor(0.0f, 0.0f, 0.05f, 0.0f);

	}

	private void uninitialize(){


		if(vbo_Sphere_Element_RRJ[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Sphere_Element_RRJ, 0);
			vbo_Sphere_Element_RRJ[0] = 0;
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

		float modelViewMatrix_RRJ[] = new float[4*4];
		float viewMatrix_RRJ[] = new float[4*4];


		GLES32.glClear(GLES32.GL_COLOR_BUFFER_BIT | GLES32.GL_DEPTH_BUFFER_BIT);


		GLES32.glUseProgram(shaderProgramObject_RRJ);


			//Sun

			Matrix.setIdentityM(viewMatrix_RRJ, 0);
			Matrix.setIdentityM(modelViewMatrix_RRJ, 0);


			Matrix.setLookAtM(viewMatrix_RRJ, 0,
							0.0f, 0.0f, 5.0f,
							0.0f, 0.0f, 0.0f,
							0.0f, 1.0f, 0.0f);



			Matrix.multiplyMM(modelViewMatrix_RRJ, 0, viewMatrix_RRJ, 0, modelViewMatrix_RRJ, 0);

			my_glPushMatrix(modelViewMatrix_RRJ);


				GLES32.glUniformMatrix4fv(modelViewUniform_RRJ, 1, false, modelViewMatrix_RRJ, 0);
				GLES32.glUniformMatrix4fv(projectionMatrixUniform_RRJ, 1, false, perspectiveProjectionMatrix_RRJ, 0);


				GLES32.glBindVertexArray(vao_Sphere_RRJ[0]);
					GLES32.glVertexAttrib3f(GLESMacros.AMC_ATTRIBUTE_COLOR, 1.0f, 1.0f, 0.0f);

					GLES32.glBindBuffer(GLES32.GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element_RRJ[0]);
					GLES32.glDrawElements(GLES32.GL_TRIANGLES, numElements_RRJ, GLES32.GL_UNSIGNED_SHORT, 0);

				GLES32.glBindVertexArray(0);

			

			//Earth
			modelViewMatrix_RRJ = null;
			modelViewMatrix_RRJ = my_glPopMatrix();


			Matrix.rotateM(modelViewMatrix_RRJ, 0, year_RRJ, 0.0f, 1.0f, 0.0f);
			Matrix.translateM(modelViewMatrix_RRJ, 0, 2.0f, 0.0f, 0.0f);


			my_glPushMatrix(modelViewMatrix_RRJ);


			Matrix.scaleM(modelViewMatrix_RRJ, 0, 0.5f, 0.5f, 0.5f);
			Matrix.rotateM(modelViewMatrix_RRJ, 0, day_RRJ, 0.0f, 1.0f, 0.0f);


	

				GLES32.glUniformMatrix4fv(modelViewUniform_RRJ, 1, false, modelViewMatrix_RRJ, 0);
				GLES32.glUniformMatrix4fv(projectionMatrixUniform_RRJ, 1, false, perspectiveProjectionMatrix_RRJ, 0);


				GLES32.glBindVertexArray(vao_Sphere_RRJ[0]);
					GLES32.glVertexAttrib3f(GLESMacros.AMC_ATTRIBUTE_COLOR, 0.40f, 0.90f, 1.0f);

					GLES32.glBindBuffer(GLES32.GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element_RRJ[0]);
					GLES32.glDrawElements(GLES32.GL_LINES, numElements_RRJ, GLES32.GL_UNSIGNED_SHORT, 0);

				GLES32.glBindVertexArray(0);


			

			//Moon

			modelViewMatrix_RRJ = null;
			modelViewMatrix_RRJ = my_glPopMatrix();	

				
			Matrix.rotateM(modelViewMatrix_RRJ, 0, moon_RRJ, 0.0f, 1.0f, 0.0f);
			Matrix.translateM(modelViewMatrix_RRJ, 0, 1.0f, 0.0f, 0.0f);


			my_glPushMatrix(modelViewMatrix_RRJ);

			Matrix.scaleM(modelViewMatrix_RRJ, 0, 0.2f, 0.2f, 0.2f);


			GLES32.glUniformMatrix4fv(modelViewUniform_RRJ, 1, false, modelViewMatrix_RRJ, 0);
			GLES32.glUniformMatrix4fv(projectionMatrixUniform_RRJ, 1, false, perspectiveProjectionMatrix_RRJ, 0);

			GLES32.glBindVertexArray(vao_Sphere_RRJ[0]);

				GLES32.glVertexAttrib3f(GLESMacros.AMC_ATTRIBUTE_COLOR, 0.5f, 0.5f, 0.5f);

				GLES32.glBindBuffer(GLES32.GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element_RRJ[0]);
				GLES32.glDrawElements(GLES32.GL_TRIANGLES, numElements_RRJ, GLES32.GL_UNSIGNED_SHORT, 0);

			GLES32.glBindVertexArray(0);

			my_glPopMatrix();
			

		GLES32.glUseProgram(0);

		requestRender();

	}

	private void update(){

		if(iWhichRotation_RRJ == YEAR_RRJ){
			if(iWhichDirection_RRJ == CLKWISE_RRJ)
				year_RRJ = (year_RRJ + 1) % 360;
			else
				year_RRJ = (year_RRJ - 1) % 360;
		}
		else if(iWhichRotation_RRJ == DAY_RRJ){
			if(iWhichDirection_RRJ == CLKWISE_RRJ)
				day_RRJ = (day_RRJ + 1) % 360;
			else
				day_RRJ = (day_RRJ - 1) % 360;
			
		}
		else if(iWhichRotation_RRJ == MOON_RRJ){
			if(iWhichDirection_RRJ == CLKWISE_RRJ)
				moon_RRJ = (moon_RRJ + 1) % 360;
			else
				moon_RRJ = (moon_RRJ - 1) % 360;
	
		}

	}

	private void my_glPushMatrix(float matrix[]){

		if(iStackTop_RRJ > 32){
			System.out.println("RTR: Stack Overflow!!");
			uninitialize();
			System.exit(1);
		}
		
		for(int i = 0; i < 16; i++){
			my_ModelViewMatrixStack_RRJ[(iStackTop_RRJ * 16) + i] = matrix[i];
		}

		iStackTop_RRJ = iStackTop_RRJ + 1;
	}

	private float[] my_glPopMatrix(){

		float temp_RRJ[] = new float[4*4];

		iStackTop_RRJ = iStackTop_RRJ - 1;

		if(iStackTop_RRJ < 0){
			iStackTop_RRJ = 0;
			return(null);
		}

		for(int i = 0; i < 16; i++){
			temp_RRJ[i] = my_ModelViewMatrixStack_RRJ[(iStackTop_RRJ * 16) + i];
		}

		return(temp_RRJ);
	}

}
