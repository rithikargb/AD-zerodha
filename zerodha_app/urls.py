from django.urls import path
from . import views
urlpatterns = [
    path('',views.home,name='home'),
    path('stock-graph/<str:symbol>/<str:start>/<str:end>', views.generate_stock_graph, name='stock_graph'),
    path('chatbot/<str:question>/',views.chatBot,name='chatbot')
]