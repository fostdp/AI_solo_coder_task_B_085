from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .models import PatinaShaderRenderer, PatinaShaderState, polish_level


class PatinaShaderView(APIView):
    def get(self, request):
        renderer = PatinaShaderRenderer()
        shaders = renderer.get_shader_sources()
        uniforms = renderer.get_uniforms()

        return Response({
            'shaders': shaders,
            'uniforms': uniforms,
            'state': {
                'polish_level': renderer._state.polish_level,
                'click_count': renderer._state.click_count,
                'light_dir': list(renderer._state.light_dir),
                'view_dir': list(renderer._state.view_dir),
                'base_color': list(renderer._state.base_color)
            },
            'formula': {
                'polish_level': '1 - exp(-click_count / 800.0)',
                'effective_shininess': '48.0 * (0.3 + polishLevel * 0.7)',
                'ambient': '0.25',
                'diffuse': 'max(dot(N, L), 0.0) * 0.6',
                'specular': 'pow(max(dot(N, H), 0.0), effectiveShininess) * (0.15 + polishLevel * 0.6)'
            }
        })

    def post(self, request):
        try:
            click_count = int(request.data.get('click_count', 0))
            light_dir = request.data.get('light_dir')
            view_dir = request.data.get('view_dir')
            base_color = request.data.get('base_color')

            renderer = PatinaShaderRenderer()

            state = PatinaShaderState(
                click_count=click_count,
                polish_level=polish_level(click_count)
            )

            if light_dir:
                state.light_dir = tuple(float(x) for x in light_dir)
            if view_dir:
                state.view_dir = tuple(float(x) for x in view_dir)
            if base_color:
                state.base_color = tuple(float(x) for x in base_color)

            uniforms = renderer.get_uniforms(state)
            shaders = renderer.get_shader_sources()

            return Response({
                'shaders': shaders,
                'uniforms': uniforms,
                'state': {
                    'polish_level': state.polish_level,
                    'click_count': state.click_count,
                    'light_dir': list(state.light_dir),
                    'view_dir': list(state.view_dir),
                    'base_color': list(state.base_color)
                },
                'lighting': {
                    'effective_shininess': renderer.compute_effective_shininess(state.polish_level),
                    'specular_strength': renderer.compute_specular_strength(state.polish_level)
                }
            })
        except (ValueError, TypeError) as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
