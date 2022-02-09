// On load function
var g_canvas = null;
var g_context = null;

function main()
{
	// Get <cnavas> element 
	g_canvas = document.getElementById("AMC");
	if (!g_canvas)
		console.log("Obtaining canvas failed\n");
	else
		console.log("Obtaining canvas succeeded");
	
	//	Print convas width and height
	console.log("canvas width "+g_canvas.width+" and canvas height "+g_canvas.height);
	
	//	Get 2D Context
	g_context = g_canvas.getContext("2d");
	if (!g_context)
		console.log("Obtaining 2d context failed\n");
	else
		console.log("Obtaining 2d context succeeded\n");
	
	//	Fill canvas with black color 
	g_context.fillStyle = "black";//	"#0000000"
	g_context.fillRect(0, 0, g_canvas.width, g_canvas.height);
	
	//	Draw text
	drawText("Hello World !!!");
	
	//	Register keyboards keydown event handler 
	window.addEventListener("keydown", keyDown, false);
	window.addEventListener("click", mouseDown, false);
}


function drawText(text)
{
	//	Center the text
	g_context.textAlign = "center";	//	Center horizontally
	g_context.textBaseline = "middle"; //	Center vertically
	
	// Text font
	g_context.font = "48px sans-serif"; // "#FFFFFF"
	
	//	Text color 
	g_context.fillStyle = "white";
	
	//	display the text in center 
	g_context.fillText(text, g_canvas.width/2, g_canvas.height/2);
}

function toggleFullscreen()
{
	//	Code
	var fullscreen_element = document.fullscreenElement				||
							 document.webkitFullscreenElement		||	//	Apple browser
							 document.mozFullScreenElement			||	//	Mozilla browser
							 document.msFullscreenElement			||	//	Microsoft (IE)
							 null;
							 
	if (null == fullscreen_element)
	{
		if (g_canvas.requestFullscreen)
			g_canvas.requestFullscreen();
		else if (g_canvas.mozRequestFullScreen)
			g_canvas.mozRequestFullScreen();
		else if (g_canvas.webkitRequestFullscreen)
			g_canvas.webkitRequestFullscreen();
		else if (g_canvas.msRequestFullscreen)
			g_canvas.msRequestFullscreen()
	}
	else	//	If already fullscreen 
	{
		if (document.exitFullscreen)
			document.exitFullscreen();
		else if (document.mozCancelFullScreen)
			document.mozCancelFullScreen();
		else if (document.webkitExitFullscreen)
			document.webkitExitFullscreen();
		else if (document.msExitFullscreen)
			document.msExitFullscreen();
	}
}

function keyDown(event)
{
	switch(event.keyCode)
	{
		case 70:	//	for 'F' or 'f'
			toggleFullscreen();
			
			//	Repaint
			drawText("Hello World !!!");
			break;
			
		case 27:	// Escape
			window.close();	//	May not work in firefox but works in safari and chrome.
			break;
	}
}

function mouseDown()
{
	
}