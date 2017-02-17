precision mediump float;

varying vec2 tcoord;
uniform sampler2D tex;
uniform vec4 col;

void main(void) 
{
    gl_FragColor = col*texture2D(tex,tcoord);
}
