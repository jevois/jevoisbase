// From here: http://mmmovania.blogspot.com/2011/11/twirl-filter-in-glsl.html

precision mediump float;

varying vec2 tcoord;
uniform sampler2D tex;
uniform float twirlamount;

void main()
{
   vec2 uv = tcoord + vec2(-0.5, -0.5);
   float angle = atan(uv.y, uv.x);
   float radius = length(uv);
   angle += radius * twirlamount;
   vec2 shifted = radius * vec2(cos(angle), sin(angle));
   gl_FragColor = texture2D(tex, (shifted + 0.5));
}
