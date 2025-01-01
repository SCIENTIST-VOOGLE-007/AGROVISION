from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('graph-operations/', views.graph_operations, name='graph_operations'),
]
