

// @author Ali AlSaibie
// @email ali\@alsaibie.com

// Adopted from Vegard Øye found at https://epsil@bitbucket.org/mpg_papers/thesis-2015-vegard.git
// This is the 4th strategy in the paper (GDBB). The vertex shader is a pass through shader and the 
// fragment shader performs backward mapping. The process is as follows:
// Tex to Position Transformation -> Cartesian to Polar -> Fisheye Backward Mapping -> Polar to Cartesian -> Back 
// to Tex and clamped.


precision mediump float;

varying vec2 tcoord;
uniform sampler2D tex;

// Clamping the transformed color to within [0,1]

vec4 color(sampler2D texture, vec2 pos) {
    if(pos.x < 0.0 || pos.y < 0.0 || pos.x > 1.0 || pos.y > 1.0) {
        return vec4(1.0, 1.0, 1.0, 1.0); // white
    } else {
        return texture2D(texture, pos);
    }
}

// Transforming tex coordinate to position

vec2 texcoordtopos(vec2 _tex) {
    return 2.0 * vec2(_tex.x - 0.5, _tex.y - 0.5);
}

// Transfering position coordinate back to tex

vec2 postotexcoord(vec2 pos) {
    return vec2(pos.x / 2.0 + 0.5, pos.y / 2.0 + 0.5);
}

// Cartesian to Polar Coordinates

vec2 toPolar(vec2 point) {
    float r = length(point);
    float theta = atan(point.y, point.x);
    return vec2(r, theta);
}

// Polar to Cartesian Coordinates

vec2 toPoint(vec2 pol) {
    float r = pol.x;
    float theta = pol.y;
    float x = r * cos(theta);
    float y = r * sin(theta);
    return vec2(x, y);
}

//  Mapping fisheye coordinates using a logarithmic model, which is faster than a polynominal model
//  for backward mapping.
//  Pixels are represented in polar coordinates. The model relates the distorted radius rd to 
//  the undistorted radius ru. 

vec2 fisheye(vec2 pos) {
    vec2 p = toPolar(pos);
    float rd = p.x;
    float theta = p.y;
    float s = 1.1;
    float lambda = 1.8;
    float ru = s * log(1.0 + lambda * rd);
    vec2 pp = vec2(ru, theta);
    return toPoint(pp);
}

vec2 transform(vec2 _tex) {
    return postotexcoord(fisheye(texcoordtopos(_tex)));
}

void main() {
    gl_FragColor = color(tex, transform(tcoord));
}