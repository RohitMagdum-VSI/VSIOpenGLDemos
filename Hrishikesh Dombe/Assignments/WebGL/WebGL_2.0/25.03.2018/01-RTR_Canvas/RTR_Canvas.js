//onload function
function main()
{
    //get <canvas> element
    var canvas = document.getElementById("HAD");
    if(!canvas)
        console.log("Obtaining Canvas Failed\n");
    else 
        console.log("Obtaining Canvas Succeeded\n");

    //print canvas width and height on console
    console.log("Canvas Width :"+canvas.width+"And Canvas Height : "+canvas.height);

    //get 2D context
    var context = canvas.getContext("2d");
    if(!context)
        console.log("Obtaining 2D Context Failed\n");
    else
        console.log("Obtaining 2D Context Succeeded\n");

    //fill canvas with black color
    context.fillStyle="black";
    context.fillRect(0,0,canvas.width,canvas.height);

    //Center the text
    context.textAlign="center";
    context.textBaseline="middle";

    //text
    var str="Hello World !!!";

    //text font
    context.font="48px sans-serif";

    //text color
    context.fillStyle="White";

    //display text in center
    context.fillText(str,canvas.width/2,canvas.height/2);
}