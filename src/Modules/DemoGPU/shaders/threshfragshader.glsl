precision mediump float;

varying vec2 tcoord;
uniform sampler2D tex;
uniform vec4 col;

void main(void) 
{
	vec4 texcol = texture2D(tex,tcoord);
    gl_FragColor = vec4(greaterThanEqual(texcol,col));
}
