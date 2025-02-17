from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.index, name='home'),
    path('tryon/', views.try_on, name='tryon'),
    path('contact/', views.contact, name='contact'),
    path('about/', views.about, name='about'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('upload/', views.upload_image, name='upload_image'),
    path('upload_image/', views.upload_image, name='upload_image_ajax'),
    path('webcam/', views.webcam_try_on, name='webcam_try_on'),
    path('stop_camera/', views.stop_camera, name='stop_camera'),
    path('reset_camera/', views.reset_camera, name='reset_camera'),
]

# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

