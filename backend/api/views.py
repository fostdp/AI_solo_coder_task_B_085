from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from datetime import datetime
import numpy as np

from .mongodb import get_collection


class JadeArtifactList(APIView):
    def get(self, request):
        collection = get_collection('jade_artifacts')
        page = int(request.GET.get('page', 1))
        page_size = int(request.GET.get('page_size', 20))
        culture = request.GET.get('culture', '')
        keyword = request.GET.get('keyword', '')
        site_name = request.GET.get('site_name', '')

        query = {}
        if culture:
            query['culture'] = culture
        if keyword:
            query['name'] = {'$regex': keyword}
        if site_name:
            query['location.site_name'] = site_name

        total = collection.count_documents(query)
        artifacts = list(collection.find(query).skip((page - 1) * page_size).limit(page_size))

        for a in artifacts:
            a['_id'] = str(a['_id'])

        return Response({
            'total': total,
            'page': page,
            'page_size': page_size,
            'data': artifacts
        })


class JadeArtifactDetail(APIView):
    def get(self, request, artifact_id):
        collection = get_collection('jade_artifacts')
        artifact = collection.find_one({'artifact_id': artifact_id})
        if artifact:
            artifact['_id'] = str(artifact['_id'])
            return Response(artifact)
        return Response({'error': 'Not found'}, status=status.HTTP_404_NOT_FOUND)


class JadeGeoSearch(APIView):
    def get(self, request):
        collection = get_collection('jade_artifacts')

        lng = request.GET.get('lng')
        lat = request.GET.get('lat')
        max_dist = float(request.GET.get('max_distance_m', 5000))
        culture = request.GET.get('culture', '')
        site_name = request.GET.get('site_name', '')

        if not lng or not lat:
            site_defaults = {
                '牛河梁': [119.3870, 41.3190],
                '反山': [120.0000, 30.3950],
                '良渚': [120.0000, 30.3950],
                '红山': [119.3870, 41.3190]
            }
            if site_name and site_name in site_defaults:
                lng, lat = site_defaults[site_name]
            elif culture == '良渚文化':
                lng, lat = 120.0000, 30.3950
            else:
                lng, lat = 119.3870, 41.3190

        try:
            lng = float(lng)
            lat = float(lat)
        except (TypeError, ValueError):
            return Response({'error': 'Invalid lng/lat'}, status=400)

        pipeline = [
            {
                '$geoNear': {
                    'near': {'type': 'Point', 'coordinates': [lng, lat]},
                    'distanceField': 'distance_m',
                    'maxDistance': max_dist,
                    'spherical': True,
                    'query': {'location.pit': {'$exists': True}}
                }
            }
        ]

        if culture:
            pipeline[0]['$geoNear']['query']['culture'] = culture

        if site_name:
            pipeline[0]['$geoNear']['query']['location.site_name'] = {
                '$regex': site_name
            }

        pipeline.extend([
            {'$limit': 100},
            {
                '$lookup': {
                    'from': 'anomaly_results',
                    'let': {'aid': '$artifact_id'},
                    'pipeline': [
                        {'$match': {'$expr': {'$eq': ['$artifact_id', '$$aid']}}},
                        {'$sort': {'timestamp': -1}},
                        {'$limit': 1}
                    ],
                    'as': 'anomaly_latest'
                }
            }
        ])

        results = list(collection.aggregate(pipeline, allowDiskUse=True))

        for a in results:
            a['_id'] = str(a['_id'])
            if a.get('anomaly_latest'):
                a['forgery_probability'] = a['anomaly_latest'][0].get('forgery_probability')
            del a['anomaly_latest']

        return Response({
            'query': {
                'center': {'lng': lng, 'lat': lat},
                'max_distance_m': max_dist,
                'culture': culture or 'all',
                'site': site_name or 'all'
            },
            'total': len(results),
            'data': results
        })


class DiffusionTensorView(APIView):
    def get(self, request, artifact_id):
        artifact_coll = get_collection('jade_artifacts')
        artifact = artifact_coll.find_one({'artifact_id': artifact_id})
        if not artifact:
            return Response({'error': 'Not found'}, status=404)

        from diffusion_solver.tasks import _build_diffusion_model

        model = _build_diffusion_model(artifact, use_anisotropic=True)

        fe_dir_map = model.get_directional_diffusivity_map(
            'Fe3+', temperature_c=25.0
        )
        mn_dir_map = model.get_directional_diffusivity_map(
            'Mn2+', temperature_c=25.0
        )

        return Response({
            'artifact_id': artifact_id,
            'jade_culture': artifact.get('culture'),
            'jade_type': artifact.get('jade_type'),
            'texture': artifact.get('texture'),
            'tensor_info': model.get_tensor_info(),
            'directional_diffusivity': {
                'Fe3+': fe_dir_map,
                'Mn2+': mn_dir_map
            }
        })

    def post(self, request, artifact_id):
        artifact_coll = get_collection('jade_artifacts')
        artifact = artifact_coll.find_one({'artifact_id': artifact_id})
        if not artifact:
            return Response({'error': 'Not found'}, status=404)

        temperature = request.data.get('temperature', 25)
        humidity = request.data.get('humidity', 50)
        time_hours = request.data.get('time_hours', 5000)

        from diffusion_solver.tasks import solve_tensor_comparison

        solve_tensor_comparison.delay(artifact_id, temperature, humidity, time_hours)

        return Response({
            'status': 'submitted',
            'task_type': 'diffusion_tensor',
            'artifact_id': artifact_id,
            'temperature': temperature,
            'humidity': humidity,
            'time_hours': time_hours,
        })


class SpectrumDataView(APIView):
    def get(self, request, artifact_id):
        collection = get_collection('spectrum_data')
        limit = int(request.GET.get('limit', 100))
        start_time = request.GET.get('start_time', '')
        end_time = request.GET.get('end_time', '')

        query = {'artifact_id': artifact_id}
        if start_time:
            query['timestamp'] = {'$gte': datetime.fromisoformat(start_time)}
        if end_time:
            if 'timestamp' in query:
                query['timestamp']['$lte'] = datetime.fromisoformat(end_time)
            else:
                query['timestamp'] = {'$lte': datetime.fromisoformat(end_time)}

        data = list(collection.find(query).sort('timestamp', -1).limit(limit))
        for d in data:
            d['_id'] = str(d['_id'])

        return Response({'data': data})


class RamanSpectrumView(APIView):
    def get(self, request, artifact_id):
        collection = get_collection('raman_spectrum')
        latest = collection.find_one(
            {'artifact_id': artifact_id},
            sort=[('timestamp', -1)]
        )
        if latest:
            latest['_id'] = str(latest['_id'])
            return Response(latest)
        return Response({'error': 'No data'}, status=status.HTTP_404_NOT_FOUND)


class XRFSpectrumView(APIView):
    def get(self, request, artifact_id):
        collection = get_collection('xrf_spectrum')
        latest = collection.find_one(
            {'artifact_id': artifact_id},
            sort=[('timestamp', -1)]
        )
        if latest:
            latest['_id'] = str(latest['_id'])
            return Response(latest)
        return Response({'error': 'No data'}, status=status.HTTP_404_NOT_FOUND)


class DiffusionResultView(APIView):
    def get(self, request, artifact_id):
        collection = get_collection('diffusion_results')
        limit = int(request.GET.get('limit', 10))
        data = list(collection.find(
            {'artifact_id': artifact_id}
        ).sort('timestamp', -1).limit(limit))

        for d in data:
            d['_id'] = str(d['_id'])

        return Response({'data': data})

    def post(self, request, artifact_id):
        artifact_coll = get_collection('jade_artifacts')
        artifact = artifact_coll.find_one({'artifact_id': artifact_id})
        if not artifact:
            return Response({'error': 'Artifact not found'}, status=404)

        temperature = request.data.get('temperature', 25)
        humidity = request.data.get('humidity', 50)
        time_hours = request.data.get('time_hours', 1000)
        use_anisotropic = bool(request.data.get('use_anisotropic', True))

        from diffusion_solver.tasks import solve_diffusion

        solve_diffusion.delay(artifact_id, temperature, humidity, time_hours, use_anisotropic)

        return Response({
            'status': 'submitted',
            'task_type': 'diffusion',
            'artifact_id': artifact_id,
        })


class AnomalyResultView(APIView):
    def get(self, request, artifact_id):
        collection = get_collection('anomaly_results')
        limit = int(request.GET.get('limit', 10))
        data = list(collection.find(
            {'artifact_id': artifact_id}
        ).sort('timestamp', -1).limit(limit))

        for d in data:
            d['_id'] = str(d['_id'])

        return Response({'data': data})

    def post(self, request, artifact_id):
        artifact_coll = get_collection('jade_artifacts')
        artifact = artifact_coll.find_one({'artifact_id': artifact_id})
        if not artifact:
            return Response({'error': 'Artifact not found'}, status=404)

        from anomaly_detector.tasks import detect_anomaly

        detect_anomaly.delay(artifact_id)

        return Response({
            'status': 'submitted',
            'task_type': 'anomaly',
            'artifact_id': artifact_id,
        })


class DensityMapView(APIView):
    def get(self, request, artifact_id):
        diffusion_coll = get_collection('diffusion_results')
        artifact_coll = get_collection('jade_artifacts')

        diffusion = diffusion_coll.find_one(
            {'artifact_id': artifact_id},
            sort=[('timestamp', -1)]
        )
        artifact = artifact_coll.find_one({'artifact_id': artifact_id})

        if not diffusion or not artifact:
            return Response({'error': 'No data'}, status=404)

        width = artifact.get('size', {}).get('width', 50)
        height = artifact.get('size', {}).get('length', 80)

        fe_profile = np.array(diffusion['fe3_diffusion']['concentration_profile'])
        mn_profile = np.array(diffusion['mn2_diffusion']['concentration_profile'])

        grid_size = 100
        density_map = np.zeros((grid_size, grid_size))

        center_x = grid_size // 2
        center_y = grid_size // 2

        tensor_info = diffusion.get('tensor_info', {})
        anisotropy_ratio = tensor_info.get('anisotropy_ratio', 1.0)

        for i in range(grid_size):
            for j in range(grid_size):
                dx = j - center_x
                dy = i - center_y
                dist = np.sqrt(dx * dx + dy * dy)
                angle = np.arctan2(dy, dx)

                direction_factor = 1.0 + 0.12 * (
                    anisotropy_ratio - 1.0
                ) * np.cos(2 * angle)

                normalized_dist = dist / (grid_size / 2)
                normalized_dist *= direction_factor

                if normalized_dist < len(fe_profile) / grid_size:
                    idx = int(normalized_dist * grid_size)
                    if idx < len(fe_profile) and idx < len(mn_profile):
                        density_map[i, j] = (
                            fe_profile[min(idx, len(fe_profile) - 1)] * 0.6
                            + mn_profile[min(idx, len(mn_profile) - 1)] * 0.4
                        )

        max_val = density_map.max() if density_map.max() > 0 else 1
        density_map_normalized = (density_map / max_val * 255).astype(int)

        return Response({
            'artifact_id': artifact_id,
            'grid_size': grid_size,
            'density_map': density_map_normalized.tolist(),
            'max_concentration': float(max_val),
            'width_mm': width,
            'height_mm': height,
            'anisotropy_applied': anisotropy_ratio > 1.01,
            'anisotropy_ratio': anisotropy_ratio
        })


class SpectrumUpload(APIView):
    def post(self, request):
        data = request.data
        artifact_id = data.get('artifact_id')
        device_id = data.get('device_id')
        spectrum_type = data.get('type', 'raman')
        spectrum_data = data.get('spectrum_data', [])
        wavelengths = data.get('wavelengths', [])
        energies = data.get('energies', [])

        if not artifact_id or not spectrum_data:
            return Response({'error': 'Missing required fields'}, status=400)

        from fiveg_receiver.tasks import receive_spectrum

        receive_spectrum.delay(
            artifact_id=artifact_id,
            device_id=device_id,
            spectrum_type=spectrum_type,
            spectrum_data=spectrum_data,
            wavelengths=wavelengths,
            energies=energies,
        )

        return Response({'status': 'submitted', 'artifact_id': artifact_id})


class StatsSummary(APIView):
    def get(self, request):
        artifact_coll = get_collection('jade_artifacts')
        alert_coll = get_collection('alerts')
        device_coll = get_collection('devices')
        anomaly_coll = get_collection('anomaly_results')

        total_artifacts = artifact_coll.count_documents({})
        hongshan_count = artifact_coll.count_documents({'culture': '红山文化'})
        liangzhu_count = artifact_coll.count_documents({'culture': '良渚文化'})

        active_alerts = alert_coll.count_documents({'status': 'active'})
        total_alerts = alert_coll.count_documents({})

        devices_online = device_coll.count_documents({'status': 'online'})
        total_devices = device_coll.count_documents({})

        spatial_count = artifact_coll.count_documents({'location.pit': {'$exists': True}})
        aniso_enabled_count = artifact_coll.count_documents({'texture': {'$exists': True}})

        anomalies = list(anomaly_coll.find({}).sort('timestamp', -1).limit(200))
        high_risk = sum(1 for a in anomalies if a.get('forgery_probability', 0) > 0.7)

        return Response({
            'total_artifacts': total_artifacts,
            'hongshan_count': hongshan_count,
            'liangzhu_count': liangzhu_count,
            'active_alerts': active_alerts,
            'total_alerts': total_alerts,
            'devices_online': devices_online,
            'total_devices': total_devices,
            'high_risk_artifacts': high_risk,
            'spatial_indexed_artifacts': spatial_count,
            'anisotropy_calibrated': aniso_enabled_count,
            'last_update': datetime.now().isoformat()
        })


class SimulatorStart(APIView):
    def post(self, request):
        from fiveg_receiver.views import SimulatorStartView
        return SimulatorStartView().post(request)


class SimulatorStop(APIView):
    def post(self, request):
        from fiveg_receiver.views import SimulatorStopView
        return SimulatorStopView().post(request)


class ProvenanceView(APIView):
    def get(self, request, artifact_id):
        collection = get_collection('provenance_results')
        limit = int(request.GET.get('limit', 10))
        data = list(collection.find(
            {'artifact_id': artifact_id}
        ).sort('timestamp', -1).limit(limit))

        for d in data:
            d['_id'] = str(d['_id'])

        return Response({'data': data})

    def post(self, request, artifact_id):
        artifact_coll = get_collection('jade_artifacts')
        artifact = artifact_coll.find_one({'artifact_id': artifact_id})
        if not artifact:
            return Response({'error': 'Artifact not found'}, status=404)

        from provenance_rf.tasks import trace_provenance

        xrf_data = request.data.get('xrf_spectrum')
        trace_provenance.delay(artifact_id, xrf_data)

        return Response({
            'status': 'submitted',
            'task_type': 'provenance_trace',
            'artifact_id': artifact_id,
        })


class PHInversionView(APIView):
    def get(self, request, artifact_id):
        collection = get_collection('ph_inversion_results')
        limit = int(request.GET.get('limit', 10))
        data = list(collection.find(
            {'artifact_id': artifact_id}
        ).sort('timestamp', -1).limit(limit))

        for d in data:
            d['_id'] = str(d['_id'])

        return Response({'data': data})

    def post(self, request, artifact_id):
        artifact_coll = get_collection('jade_artifacts')
        artifact = artifact_coll.find_one({'artifact_id': artifact_id})
        if not artifact:
            return Response({'error': 'Artifact not found'}, status=404)

        from ph_inversion.tasks import invert_ph_history

        diffusion_data = request.data.get('diffusion_result')
        invert_ph_history.delay(artifact_id, diffusion_data)

        return Response({
            'status': 'submitted',
            'task_type': 'ph_inversion',
            'artifact_id': artifact_id,
        })


class ForgeryProcessView(APIView):
    def get(self, request, artifact_id):
        collection = get_collection('forgery_classification_results')
        limit = int(request.GET.get('limit', 10))
        data = list(collection.find(
            {'artifact_id': artifact_id}
        ).sort('timestamp', -1).limit(limit))

        for d in data:
            d['_id'] = str(d['_id'])

        return Response({'data': data})

    def post(self, request, artifact_id):
        artifact_coll = get_collection('jade_artifacts')
        artifact = artifact_coll.find_one({'artifact_id': artifact_id})
        if not artifact:
            return Response({'error': 'Artifact not found'}, status=404)

        from craft_svm.tasks import classify_forgery_process

        raman_data = request.data.get('raman_spectrum')
        classify_forgery_process.delay(artifact_id, raman_data)

        return Response({
            'status': 'submitted',
            'task_type': 'forgery_process',
            'artifact_id': artifact_id,
        })
