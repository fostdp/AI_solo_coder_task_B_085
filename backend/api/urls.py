from django.urls import path, include
from . import views
from fiveg_receiver import urls as receiver_urls
from alert_ws import urls as alert_urls

urlpatterns = [
    path('artifacts/', views.JadeArtifactList.as_view()),
    path('artifacts/geo-search/', views.JadeGeoSearch.as_view()),
    path('artifacts/<str:artifact_id>/', views.JadeArtifactDetail.as_view()),
    path('artifacts/<str:artifact_id>/spectrum/', views.SpectrumDataView.as_view()),
    path('artifacts/<str:artifact_id>/raman/', views.RamanSpectrumView.as_view()),
    path('artifacts/<str:artifact_id>/xrf/', views.XRFSpectrumView.as_view()),
    path('artifacts/<str:artifact_id>/diffusion/', views.DiffusionResultView.as_view()),
    path('artifacts/<str:artifact_id>/diffusion-tensor/', views.DiffusionTensorView.as_view()),
    path('artifacts/<str:artifact_id>/anomaly/', views.AnomalyResultView.as_view()),
    path('artifacts/<str:artifact_id>/density-map/', views.DensityMapView.as_view()),
    path('artifacts/<str:artifact_id>/provenance/', views.ProvenanceView.as_view()),
    path('artifacts/<str:artifact_id>/ph-inversion/', views.PHInversionView.as_view()),
    path('artifacts/<str:artifact_id>/forgery-process/', views.ForgeryProcessView.as_view()),
    path('', include(receiver_urls)),
    path('', include(alert_urls)),
    path('stats/summary/', views.StatsSummary.as_view()),
]
