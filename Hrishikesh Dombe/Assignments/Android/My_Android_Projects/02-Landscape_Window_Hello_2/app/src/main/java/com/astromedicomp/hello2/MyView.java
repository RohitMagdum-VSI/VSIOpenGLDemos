package com.astromedicomp.hello2;

import android.content.Context;
import android.app.Activity;
import android.os.Bundle;
import android.widget.TextView;
import android.view.Gravity;
import android.graphics.Color;

public class MyView extends TextView
{
	MyView(Context context)
	{
		super(context);
		setText("Hello World 2 !!!");
		setTextSize(60);
		setTextColor(Color.rgb(255,128,0));
		setGravity(Gravity.CENTER);	
	}
}