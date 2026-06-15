from django.urls import path
from . import views

urlpatterns = [
    path('artifacts/', views.JadeArtifactList.as_view(), name='artifact-list'),
    path('artifacts/geo-search/', views.JadeGeoSearch.as_view(), name='artifact-geo-search'),
    path('artifacts/<str:artifact_id>/', views.JadeArtifactDetail.as_view(), name='artifact-detail'),
    path('artifacts/<str:artifact_id>/spectrum/', views.SpectrumDataView.as_view(), name='spectrum-data'),
    path('artifacts/<str:artifact_id>/raman/', views.RamanSpectrumView.as_view(), name='raman-spectrum'),
    path('artifacts/<str:artifact_id>/xrf/', views.XRFSpectrumView.as_view(), name='xrf-spectrum'),
    path('artifacts/<str:artifact_id>/diffusion/', views.DiffusionResultView.as_view(), name='diffusion-result'),
    path('artifacts/<str:artifact_id>/diffusion-tensor/', views.DiffusionTensorView.as_view(), name='diffusion-tensor'),
    path('artifacts/<str:artifact_id>/anomaly/', views.AnomalyResultView.as_view(), name='anomaly-result'),
    path('artifacts/<str:artifact_id>/density-map/', views.DensityMapView.as_view(), name='density-map'),
    path('alerts/', views.AlertList.as_view(), name='alert-list'),
    path('alerts/<str:alert_id>/acknowledge/', views.AlertAcknowledge.as_view(), name='alert-acknowledge'),
    path('devices/', views.DeviceList.as_view(), name='device-list'),
    path('spectrum/upload/', views.SpectrumUpload.as_view(), name='spectrum-upload'),
    path('stats/summary/', views.StatsSummary.as_view(), name='stats-summary'),
    path('simulator/start/', views.SimulatorStart.as_view(), name='simulator-start'),
    path('simulator/stop/', views.SimulatorStop.as_view(), name='simulator-stop'),
]
