varying vec3 v_normal;
varying vec3 v_position;
varying vec4 v_id;
varying vec3 v_color;
varying float origin_id;
const vec3 light_position = vec3(1.0,1.0,1.0);
const vec3 ambient_color = vec3(0.07, 0.07, 0.1);
const float shininess = 64.0;
const float gamma = 2.2;
uniform float select_id;
void main()
{
    vec3 normal= normalize(v_normal);
    vec3 light_direction = normalize(light_position - v_position);
    float lambertian = max(dot(light_direction,normal), 0.0);
    vec3 color_linear = ambient_color +
                        lambertian * v_color;
    vec3 color_gamma = pow(color_linear, vec3(1.0/gamma));
    gl_FragData[0] = vec4(color_gamma, 1.0);
    gl_FragData[1] = v_id;
    if (abs(origin_id - select_id)<0.01)
        gl_FragData[0] = vec4(1.0, 0.0, 0.0, 1.0);
}