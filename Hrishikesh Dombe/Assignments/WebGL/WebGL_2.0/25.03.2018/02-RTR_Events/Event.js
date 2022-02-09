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

    //center the text
    context.textAlign="center";//center horizontally
    context.textBaseline="middle";//center vertically
    console.log("1\n");
    //text
    var str="Hello World !!!";
    console.log("2\n");
    //text font
    context.font="48px sans-serif";

    //text color
    context.fillStyle="white";

    //display the text in center
    context.fillText(str,canvas.width/2,canvas.height/2);

    //register keyboard's keydown event handler
    window.addEventListener("keydown",keyDown,false);
    window.addEventListener("click",mouseDown,false);
}

function keyDown(event)
{
    alert("A Key Is Pressed");
}

function mouseDown()
{
    alert("Mouse Is Clicked");
}