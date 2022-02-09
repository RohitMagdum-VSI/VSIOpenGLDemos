package com.astromedicomp.hello1;


import android.view.Gravity;
import android.graphics.Color;
import android.widget.TextView;
import android.view.Window;
import android.view.WindowManager;
import android.content.pm.ActivityInfo;
import android.app.Activity;
import android.os.Bundle;

public class MainActivity extends Activity {

	private TextView myTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {

	//this is done to get rid of Action Bar
	this.requestWindowFeature(Window.FEATURE_NO_TITLE);

	//This is done to make Fullscreen
	this.getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,WindowManager.LayoutParams.FLAG_FULLSCREEN);

        super.onCreate(savedInstanceState);
        //setContentView(R.layout.activity_main);

	//For setting Background Black
	this.getWindow().getDecorView().setBackgroundColor(Color.rgb(0,0,0));

	MainActivity.this.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);

	myTextView = new TextView(this);

	myTextView.setText("Hello World 1 !!!");
	myTextView.setTextSize(60);
	myTextView.setTextColor(Color.GREEN);
	myTextView.setGravity(Gravity.CENTER);

	setContentView(myTextView);
    }
	@Override
	protected void onPause()
	{
		super.onPause();
	}

	@Override
	protected void onResume()
	{
		super.onResume();
	}
}
