// On load function
function main()
{
	// Get <cnavas> element 
	var canvas = document.getElementById("AMC");
	if (!canvas)
		console.log("Obtaining canvas failed\n");
	else
		console.log("Obtaining canvas succeeded");
	
	//	Print convas width and height
	console.log("canvas width "+canvas.width+" and canvas height "+canvas.height);
	
	//	Get 2D Context
	var context = canvas.getContext("2d");
	if (!context)
		console.log("Obtaining 2d context failed\n");
	else
		console.log("Obtaining 2d context succeeded\n");
	
	//	Fill canvas with black color 
	context.fillStyle = "black";//	"#0000000"
	context.fillRect(0, 0, canvas.width, canvas.height);
	
	//	Center the text
	context.textAlign = "center";	//	Center horizontally
	context.textBaseline = "middle"; //	Center vertically
	
	//	Text
	var str = "Hello World !!!";
	
	// Text font
	context.font = "48px sans-serif"; // "#FFFFFF"
	
	//	Text color 
	context.fillStyle = "white";
	
	//	display the text in center 
	context.fillText(str, canvas.width/2, canvas.height/2);
}