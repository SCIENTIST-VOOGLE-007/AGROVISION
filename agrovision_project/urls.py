from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('agrovision/', include('agrovision_app.urls')),
    path('', RedirectView.as_view(url='agrovision/', permanent=True)),  # Redirect root URL to /agrovision/
]


