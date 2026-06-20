from django.contrib import admin
from django.urls import path
from fraud_app import views

urlpatterns = [
    path("admin/",            admin.site.urls),
    path("",                  views.login_view,         name="login"),
    path("login/",            views.login_view,         name="login"),
    path("logout/",           views.logout_view,        name="logout"),
    path("dashboard/",        views.dashboard,          name="dashboard"),
    path("transactions/",     views.transaction_list,   name="transaction_list"),
    path("upload/",           views.upload_transactions,name="upload_transactions"),
    path("predict/",          views.predict_single,     name="predict_single"),
    path("predict/history/",  views.predict_history,    name="predict_history"),
    path("alerts/",           views.alert_list,         name="alert_list"),
    path("alerts/<int:pk>/",  views.alert_detail,       name="alert_detail"),
    path("seed/",             views.seed_demo_data,     name="seed_demo_data"),
]
