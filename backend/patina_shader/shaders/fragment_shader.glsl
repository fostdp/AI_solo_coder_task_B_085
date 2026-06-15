precision mediump float;

varying vec3 vNormal;
varying vec3 vPosition;
varying vec2 vTexCoord;

uniform float uPolishLevel;
uniform vec3 uLightDir;
uniform vec3 uViewDir;
uniform vec3 uBaseColor;

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(uLightDir);
    vec3 V = normalize(uViewDir);
    vec3 H = normalize(L + V);

    float effectiveShininess = 48.0 * (0.3 + uPolishLevel * 0.7);

    float ambient = 0.25;

    float diffDot = max(dot(N, L), 0.0);
    float diffuse = diffDot * 0.6;

    float specDot = max(dot(N, H), 0.0);
    float specularStrength = 0.15 + uPolishLevel * 0.6;
    float specular = pow(specDot, effectiveShininess) * specularStrength;

    vec3 finalColor = uBaseColor * (ambient + diffuse) + vec3(specular);

    gl_FragColor = vec4(finalColor, 1.0);
}
