package com.rohit_r_jadhav.graph;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;

import android.view.Window;
import android.view.WindowManager;
import android.content.pm.ActivityInfo;
import android.view.View;



public class MainActivity extends AppCompatActivity{

	private GLESView glesView;

	@Override
	protected void onCreate(Bundle savedCurrentState){
		super.onCreate(savedCurrentState);

		this.supportRequestWindowFeature(Window.FEATURE_NO_TITLE);

		this.getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_HIDE_NAVIGATION | 
												View.SYSTEM_UI_FLAG_IMMERSIVE);


		this.getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);

		MainActivity.this.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);

		glesView = new GLESView(this);

		setContentView(glesView);
	}

	@Override
	protected void onPause(){
		super.onPause();
	}

	@Override
	protected void onResume(){
		super.onResume();
	}

}

