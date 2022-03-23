#version 450

#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform sampler2D src_tex;

layout(location = 0) in vec2 texcoord;
layout(location = 0) out vec4 color;

void main() {
    float gray = texture(src_tex, texcoord).x;
    color = vec4(gray, gray, gray, 1);
}
