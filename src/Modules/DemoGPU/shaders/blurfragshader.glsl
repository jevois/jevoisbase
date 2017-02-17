precision mediump float;

varying vec2 tcoord;
uniform sampler2D tex;
uniform vec2 texelsize;

void main(void) 
{
	vec4 col = vec4(0);
	float total_added = 0.0;
	for(int xoffset = -2; xoffset <= 2; xoffset++)
	{
		for(int yoffset = -2; yoffset <= 2; yoffset++)
		{
			vec2 offset = vec2(xoffset,yoffset);
			float prop = 1.0/(offset.x*offset.x+offset.y*offset.y+1.0);
			total_added += prop;
			col += prop*texture2D(tex,tcoord+offset*texelsize);
		}
	}
	col /= total_added;
    gl_FragColor = clamp(col,vec4(0),vec4(1));
}
