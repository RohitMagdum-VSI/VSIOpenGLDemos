//global variables
var canvas=null;
var context=null;

//onload function
function main()
{
    //get <canvas> element
    canvas = document.getElementById("HAD");
    if(!canvas)
        console.log("Obtaining Canvas Failed\n");
    else
        console.log("Obtaining Canvas Succeeded\n");
        
    //print canvas width and height on console
    console.log("Canvas Width : "+canvas.width+" And Canvas Height : "+canvas.height);

    //get 2D context
    context = canvas.getContext("2d");
    if(!context)
        console.log("Obtaining 2D Context Failed\n");
    else
        console.log("Obtaining 2D Context Succeeded\n");

    //fill canvas with black color
    context.fillStyle="black";
    context.fillRect(0,0,canvas.width,canvas.height);

    drawText("Hello World !!!");
   
    //register keyboard's keydown event handler
    window.addEventListener("keydown",keyDown,false);
    window.addEventListener("click",mouseDown,false);
}

function drawText(text)
{
     //center the text
     context.textAlign="center";//center horizontally
     context.textBaseline="middle";//center vertically
     
     //text font
     context.font="48px sans-serif";
 
     //text color
     context.fillStyle="white";
 
     //display the text in center
     context.fillText(text,canvas.width/2,canvas.height/2); 
}

function toggleFullScreen()
{
    var fullscreen_element=document.fullscreenElement||document.webkitFullscreenElement||document.mozFullScreenElement||document.msFullscreenElement||null;

    if(fullscreen_element==null)
    {
        if(canvas.requestFullscreen)
            canvas.requestFullscreen();
        else if(canvas.mozRequestFullScreen)
            canvas.mozRequestFullScreen();
        else if(canvas.webkitRequestFullscreen)
            canvas.webkitRequestFullscreen();
        else if(canvas.msRequestFullscreen)
            canvas.msRequestFullscreen();
    }

    else
    {
        console.log("5\n");
        if(document.exitFullscreen)
            document.exitFullscreen();
        else if(document.mozCancelFullScreen)
            document.mozCancelFullScreen();
        else if(document.webkitExitFullscreen)
            document.webkitExitFullscreen();
        else if(document.msExitFullscreen)
            document.msExitFullscreen();
        console.log("6\n");
    }
}

function keyDown(event)
{
    switch(event.keyCode)
    {
        case 70: //F or f
            console.log("1\n");
            toggleFullScreen();
            console.log("2\n");
            //repaint as there is no default repaint in Javascript
            drawText("Hello World !!!");
            break;
    }
}

function mouseDown()
{
    alert("Mouse Is Clicked");
}