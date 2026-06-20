from django.db import models
from django.contrib.auth.models import User

class Customer(models.Model):
    CATEGORY_CHOICES = [("Residential","Residential"),("Commercial","Commercial"),("Industrial","Industrial")]
    account_number    = models.CharField(max_length=30, unique=True)
    meter_number      = models.CharField(max_length=30, unique=True)
    customer_name     = models.CharField(max_length=200)
    customer_category = models.CharField(max_length=50, choices=CATEGORY_CHOICES, default="Residential")
    connection_type   = models.CharField(max_length=30, default="Single Phase")
    geographic_zone   = models.CharField(max_length=100, blank=True)
    created_at        = models.DateTimeField(auto_now_add=True)
    def __str__(self): return f"{self.customer_name} ({self.meter_number})"
    class Meta: ordering = ["customer_name"]

class VendingPoint(models.Model):
    STATUS_CHOICES = [("Authorized","Authorized"),("Unauthorized","Unauthorized")]
    dealer_name          = models.CharField(max_length=200)
    location             = models.CharField(max_length=200, blank=True)
    authorization_status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="Authorized")
    created_at           = models.DateTimeField(auto_now_add=True)
    def __str__(self): return f"{self.dealer_name} [{self.authorization_status}]"

class Transaction(models.Model):
    customer        = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name="transactions")
    vending_point   = models.ForeignKey(VendingPoint, on_delete=models.SET_NULL, null=True, blank=True)
    token_id        = models.CharField(max_length=50, unique=True)
    purchase_date   = models.DateTimeField()
    purchased_units = models.FloatField()
    payment_amount  = models.FloatField()
    fraud_score     = models.FloatField(null=True, blank=True)
    is_flagged      = models.BooleanField(default=False)
    created_at      = models.DateTimeField(auto_now_add=True)
    def __str__(self): return f"Token {self.token_id}"
    class Meta: ordering = ["-purchase_date"]

class MeterReading(models.Model):
    STATUS_CHOICES = [("Normal","Normal"),("Tamper","Tamper Flag"),("Bypass","Bypass Detected")]
    customer          = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name="meter_readings")
    reading_date      = models.DateTimeField()
    cumulative_kwh    = models.FloatField()
    meter_status      = models.CharField(max_length=30, choices=STATUS_CHOICES, default="Normal")
    consumption_delta = models.FloatField(null=True, blank=True)
    created_at        = models.DateTimeField(auto_now_add=True)
    def __str__(self): return f"{self.customer.meter_number} — {self.reading_date.date()}"
    class Meta: ordering = ["-reading_date"]

class FraudAlert(models.Model):
    ALERT_TYPE_CHOICES = [("Token","Token Fraud"),("Tamper","Meter Tampering"),("Vending","Vending Fraud"),("General","General Anomaly")]
    STATUS_CHOICES     = [("Pending","Pending"),("Confirmed","Confirmed Fraud"),("Dismissed","Dismissed")]
    transaction          = models.ForeignKey(Transaction, on_delete=models.CASCADE, related_name="alerts")
    customer             = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name="alerts")
    alert_type           = models.CharField(max_length=20, choices=ALERT_TYPE_CHOICES, default="General")
    fraud_probability    = models.FloatField()
    investigation_status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="Pending")
    investigation_notes  = models.TextField(blank=True)
    investigated_by      = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    created_at           = models.DateTimeField(auto_now_add=True)
    updated_at           = models.DateTimeField(auto_now=True)
    def __str__(self): return f"Alert #{self.pk} [{self.investigation_status}]"
    class Meta: ordering = ["-created_at"]
