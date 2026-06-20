from django.contrib import admin
from .models import Customer, Transaction, MeterReading, VendingPoint, FraudAlert

@admin.register(Customer)
class CustomerAdmin(admin.ModelAdmin):
    list_display  = ["meter_number", "customer_name", "customer_category", "geographic_zone"]
    search_fields = ["meter_number", "customer_name", "account_number"]

@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    list_display  = ["token_id", "customer", "purchased_units", "fraud_score", "is_flagged", "purchase_date"]
    list_filter   = ["is_flagged"]
    search_fields = ["token_id", "customer__meter_number"]

@admin.register(FraudAlert)
class FraudAlertAdmin(admin.ModelAdmin):
    list_display  = ["pk", "customer", "alert_type", "fraud_probability", "investigation_status", "created_at"]
    list_filter   = ["alert_type", "investigation_status"]

@admin.register(VendingPoint)
class VendingPointAdmin(admin.ModelAdmin):
    list_display = ["dealer_name", "authorization_status", "location"]

@admin.register(MeterReading)
class MeterReadingAdmin(admin.ModelAdmin):
    list_display = ["customer", "reading_date", "cumulative_kwh", "meter_status"]
