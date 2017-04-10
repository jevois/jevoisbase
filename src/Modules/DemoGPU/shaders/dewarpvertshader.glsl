// Identical to simplevertshader with different offset/scale

attribute vec4 vertex;
uniform vec2 offsetd;
uniform vec2 scaled;
varying vec2 tcoord;
void main(void) 
{
	vec4 pos = vertex;
	tcoord.xy = pos.xy;
	pos.xy = pos.xy*scaled+offsetd;
	gl_Position = pos;
}