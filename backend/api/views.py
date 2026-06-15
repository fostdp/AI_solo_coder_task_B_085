from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from datetime import datetime, timedelta
import numpy as np
import json

from .mongodb import get_collection
from algorithms.diffusion_model import DiffusionModel
from algorithms.diffusion.tensor import (
    DiffusionTensor,
    CTCalibratedTensorBuilder,
    get_default_tensor
)
from algorithms.isolation_forest import AnomalyDetector
from alerts.wechat_alert import WeChatAlert
from alerts.websocket_alert import WebSocketAlert

_tensor_builder = CTCalibratedTensorBuilder()
anomaly_detector = AnomalyDetector()
wechat_alert = WeChatAlert()
ws_alert = WebSocketAlert()


def _build_diffusion_model(artifact: dict, use_anisotropic: bool = True) -> DiffusionModel:
    """
    根据玉器元数据构造 DiffusionModel（绑定文化、器型、各向异性张量）
    - 优先使用 artifact.texture（CT扫描标定）
    - 否则用 jade_culture + jade_type 做预设
    """
    culture = artifact.get('culture', '红山文化')
    jade_type = artifact.get('jade_type', '玉璧')
    texture = artifact.get('texture')

    custom_tensor = None

    if texture and texture.get('main_orientation_euler'):
        try:
            euler_deg = tuple(texture['main_orientation_euler'])
            custom_tensor = _tensor_builder.build_preset(
                jade_culture=culture,
                jade_type=jade_type,
                orientation_deg=euler_deg
            )
            if texture.get('grain_size_um'):
                gs = texture['grain_size_um']
                gb_factor = 1.0 + 0.3 * (50.0 / max(gs, 1.0))
                custom_tensor.D_parallel *= gb_factor
                custom_tensor.D_perp1 *= gb_factor * 0.9
                custom_tensor.D_perp2 *= gb_factor * 0.85
        except Exception:
            custom_tensor = None

    return DiffusionModel(
        jade_culture=culture,
        jade_type=jade_type,
        use_anisotropic=use_anisotropic,
        custom_tensor=custom_tensor
    )


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
    """
    基于 2dsphere 的出土坑位地理查询接口
    GET /api/artifacts/geo-search/?lng=&lat=&max_distance_m=&culture=
    """

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
    """
    扩散张量接口
    GET  /api/artifacts/<id>/diffusion-tensor/    查询该玉器的张量参数
    POST /api/artifacts/<id>/diffusion-tensor/    运行各向异性 vs 均质对比
    """

    def get(self, request, artifact_id):
        artifact_coll = get_collection('jade_artifacts')
        artifact = artifact_coll.find_one({'artifact_id': artifact_id})
        if not artifact:
            return Response({'error': 'Not found'}, status=404)

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

        thickness = artifact.get('size', {}).get('thickness', 2.0)
        temperature = request.data.get('temperature', 25)
        humidity = request.data.get('humidity', 50)
        time_hours = request.data.get('time_hours', 5000)

        model_aniso = _build_diffusion_model(artifact, use_anisotropic=True)

        compare_fe = model_aniso.compare_isotropic_vs_anisotropic(
            'Fe3+', thickness, time_hours, temperature
        )
        compare_mn = model_aniso.compare_isotropic_vs_anisotropic(
            'Mn2+', thickness, time_hours, temperature
        )

        temp_sens = model_aniso.temperature_sensitivity_analysis(
            'Fe3+', (5, 40), thickness, time_hours
        )

        from django.conf import settings
        threshold = settings.DIFFUSION_ALERT_THRESHOLD_MM

        summary = {
            'aniso_enhancement_fe_pct': compare_fe['comparison']['relative_error_percent'],
            'aniso_enhancement_mn_pct': compare_mn['comparison']['relative_error_percent'],
            'alert_threshold_mm': threshold,
            'alert_cases': {
                'iso_triggered': (
                    compare_fe['isotropic']['depth_mm'] > threshold
                    or compare_mn['isotropic']['depth_mm'] > threshold
                ),
                'aniso_triggered': (
                    compare_fe['anisotropic']['depth_mm'] > threshold
                    or compare_mn['anisotropic']['depth_mm'] > threshold
                ),
                'discrepancy': (
                    compare_fe['comparison']['alert_discrepancy']
                    or compare_mn['comparison']['alert_discrepancy']
                )
            }
        }

        return Response({
            'artifact_id': artifact_id,
            'conditions': {
                'thickness_mm': thickness,
                'time_hours': time_hours,
                'temperature_c': temperature,
                'humidity_pct': humidity
            },
            'Fe3_comparison': compare_fe,
            'Mn2_comparison': compare_mn,
            'temperature_sensitivity_Fe3': temp_sens,
            'alert_summary': summary
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
    """
    沁色扩散模拟（修复：已切换为各向异性张量模式）
    - model 按玉器的 culture+jade_type+texture 构造
    - 超过阈值触发告警
    """

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
        collection = get_collection('diffusion_results')
        artifact_coll = get_collection('jade_artifacts')

        artifact = artifact_coll.find_one({'artifact_id': artifact_id})
        if not artifact:
            return Response({'error': 'Artifact not found'}, status=404)

        thickness = artifact.get('size', {}).get('thickness', 2.0)
        temperature = request.data.get('temperature', 25)
        humidity = request.data.get('humidity', 50)
        time_hours = request.data.get('time_hours', 1000)
        use_aniso = bool(request.data.get('use_anisotropic', True))

        model = _build_diffusion_model(artifact, use_anisotropic=use_aniso)

        fe_result = model.simulate_diffusion(
            ion_type='Fe3+',
            thickness_mm=thickness,
            time_hours=time_hours,
            temperature=temperature,
            humidity=humidity
        )

        mn_result = model.simulate_diffusion(
            ion_type='Mn2+',
            thickness_mm=thickness,
            time_hours=time_hours,
            temperature=temperature,
            humidity=humidity
        )

        penetration_depth_fe = model.calculate_penetration_depth(
            np.array(fe_result['concentration_profile']), thickness
        )
        penetration_depth_mn = model.calculate_penetration_depth(
            np.array(mn_result['concentration_profile']), thickness
        )

        tensor_info = model.get_tensor_info()

        result = {
            'artifact_id': artifact_id,
            'timestamp': datetime.now(),
            'fe3_diffusion': fe_result,
            'mn2_diffusion': mn_result,
            'penetration_depth_fe_mm': float(penetration_depth_fe),
            'penetration_depth_mn_mm': float(penetration_depth_mn),
            'max_penetration_mm': max(float(penetration_depth_fe), float(penetration_depth_mn)),
            'penetration_isotropic_fe_mm': fe_result.get(
                'penetration_depth_isotropic_mm', penetration_depth_fe
            ),
            'penetration_isotropic_mn_mm': mn_result.get(
                'penetration_depth_isotropic_mm', penetration_depth_mn
            ),
            'temperature': temperature,
            'humidity': humidity,
            'simulation_time_hours': time_hours,
            'solver_mode': fe_result.get('solver_mode', 'unknown'),
            'tensor_info': tensor_info
        }

        collection.insert_one(result)
        result['_id'] = str(result['_id'])

        from django.conf import settings
        if result['max_penetration_mm'] > settings.DIFFUSION_ALERT_THRESHOLD_MM:
            alert_data = {
                'artifact_id': artifact_id,
                'alert_type': 'diffusion',
                'severity': 'warning',
                'message': (
                    f'沁色深度超过阈值: 各向异性 {result["max_penetration_mm"]:.2f}mm'
                    f' (均质模式 ~{max(
                        result["penetration_isotropic_fe_mm"],
                        result["penetration_isotropic_mn_mm"]
                    ):.2f}mm，差异 '
                    f'{100 * (result["max_penetration_mm"] / max(
                        result["penetration_isotropic_fe_mm"],
                        result["penetration_isotropic_mn_mm"]
                    ) - 1):.1f}%)'
                ),
                'data': {
                    'penetration_mm': result['max_penetration_mm'],
                    'solver': result['solver_mode'],
                    'tensor_info': tensor_info
                },
                'timestamp': datetime.now(),
                'status': 'active'
            }
            get_collection('alerts').insert_one(alert_data)
            wechat_alert.send_alert(alert_data)
            ws_alert.broadcast_alert(alert_data)

        return Response(result)


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
        collection = get_collection('anomaly_results')
        xrf_coll = get_collection('xrf_spectrum')
        raman_coll = get_collection('raman_spectrum')

        xrf_data = xrf_coll.find_one(
            {'artifact_id': artifact_id},
            sort=[('timestamp', -1)]
        )
        raman_data = raman_coll.find_one(
            {'artifact_id': artifact_id},
            sort=[('timestamp', -1)]
        )

        if not xrf_data and not raman_data:
            return Response({'error': 'No spectrum data'}, status=400)

        features = anomaly_detector.extract_features(xrf_data, raman_data)
        result = anomaly_detector.detect(features, artifact_id)

        result_doc = {
            'artifact_id': artifact_id,
            'timestamp': datetime.now(),
            'anomaly_score': result['anomaly_score'],
            'is_anomaly': result['is_anomaly'],
            'forgery_probability': result['forgery_probability'],
            'features': result['features'],
            'anomaly_reasons': result['anomaly_reasons']
        }

        collection.insert_one(result_doc)
        result_doc['_id'] = str(result_doc['_id'])

        from django.conf import settings
        if result['forgery_probability'] > settings.ANOMALY_SCORE_THRESHOLD:
            alert_data = {
                'artifact_id': artifact_id,
                'alert_type': 'anomaly',
                'severity': 'critical',
                'message': f'检测到疑似仿古作伪，概率: {result["forgery_probability"]:.2%}',
                'data': {
                    'forgery_probability': result['forgery_probability'],
                    'anomaly_score': result['anomaly_score']
                },
                'timestamp': datetime.now(),
                'status': 'active'
            }
            get_collection('alerts').insert_one(alert_data)
            wechat_alert.send_alert(alert_data)
            ws_alert.broadcast_alert(alert_data)

        return Response(result_doc)


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


class AlertList(APIView):
    def get(self, request):
        collection = get_collection('alerts')
        status_filter = request.GET.get('status', '')
        alert_type = request.GET.get('type', '')
        limit = int(request.GET.get('limit', 50))

        query = {}
        if status_filter:
            query['status'] = status_filter
        if alert_type:
            query['alert_type'] = alert_type

        alerts = list(collection.find(query).sort('timestamp', -1).limit(limit))
        for a in alerts:
            a['_id'] = str(a['_id'])

        return Response({'data': alerts, 'total': len(alerts)})


class AlertAcknowledge(APIView):
    def post(self, request, alert_id):
        collection = get_collection('alerts')
        from bson import ObjectId

        result = collection.update_one(
            {'_id': ObjectId(alert_id)},
            {'$set': {'status': 'acknowledged', 'acknowledged_at': datetime.now()}}
        )

        if result.modified_count > 0:
            return Response({'success': True})
        return Response({'error': 'Alert not found'}, status=404)


class DeviceList(APIView):
    def get(self, request):
        collection = get_collection('devices')
        device_type = request.GET.get('type', '')

        query = {}
        if device_type:
            query['device_type'] = device_type

        devices = list(collection.find(query))
        for d in devices:
            d['_id'] = str(d['_id'])

        return Response({'data': devices, 'total': len(devices)})


class SpectrumUpload(APIView):
    def post(self, request):
        data = request.data
        artifact_id = data.get('artifact_id')
        device_id = data.get('device_id')
        spectrum_type = data.get('type', 'raman')
        spectrum_data = data.get('spectrum_data', [])

        if not artifact_id or not spectrum_data:
            return Response({'error': 'Missing required fields'}, status=400)

        timestamp = datetime.now()

        if spectrum_type == 'raman':
            collection = get_collection('raman_spectrum')
        else:
            collection = get_collection('xrf_spectrum')

        doc = {
            'artifact_id': artifact_id,
            'device_id': device_id,
            'timestamp': timestamp,
            'spectrum_data': spectrum_data,
            'wavelengths': data.get('wavelengths', [])
        }

        collection.insert_one(doc)
        doc['_id'] = str(doc['_id'])

        spectrum_coll = get_collection('spectrum_data')
        spectrum_coll.insert_one({
            'artifact_id': artifact_id,
            'device_id': device_id,
            'type': spectrum_type,
            'timestamp': timestamp
        })

        return Response({'success': True, 'data': doc})


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
        from simulator.jade_simulator import simulator
        interval = request.data.get('interval', 30)
        simulator.start(interval=interval)
        return Response({'status': 'started', 'interval': interval})


class SimulatorStop(APIView):
    def post(self, request):
        from simulator.jade_simulator import simulator
        simulator.stop()
        return Response({'status': 'stopped'})
