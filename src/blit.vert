#version 450

#extension GL_ARB_separate_shader_objects : enable

const vec3 positions[] = {
    vec3(-1, -1, 0),
    vec3(-1, 3, 0),
    vec3(3, -1, 0),
};

const vec2 texcoords[] = {
    vec2(0, 0),
    vec2(0, 2),
    vec2(2, 0),
};

layout(location = 0) out vec2 texcoord;

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 1);
    texcoord = texcoords[gl_VertexIndex];
}
