import csv, io, random, string
from datetime import timedelta
import numpy as np
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.utils import timezone
from django.db.models import Count
from .models import Customer, Transaction, MeterReading, VendingPoint, FraudAlert
from .predictor import predict

def login_view(request):
    if request.user.is_authenticated:
        return redirect("dashboard")
    if request.method == "POST":
        user = authenticate(request, username=request.POST.get("username",""), password=request.POST.get("password",""))
        if user:
            login(request, user)
            return redirect("dashboard")
        messages.error(request, "Invalid username or password.")
    return render(request, "fraud_app/login.html")

def logout_view(request):
    logout(request)
    return redirect("login")

@login_required
def dashboard(request):
    total_customers    = Customer.objects.count()
    total_transactions = Transaction.objects.count()
    total_alerts       = FraudAlert.objects.count()
    pending_alerts     = FraudAlert.objects.filter(investigation_status="Pending").count()
    confirmed_fraud    = FraudAlert.objects.filter(investigation_status="Confirmed").count()
    recent_alerts      = FraudAlert.objects.select_related("customer","transaction").order_by("-created_at")[:10]
    alert_types        = FraudAlert.objects.values("alert_type").annotate(count=Count("id")).order_by("-count")
    high_risk          = FraudAlert.objects.filter(fraud_probability__gte=0.75).count()
    medium_risk        = FraudAlert.objects.filter(fraud_probability__gte=0.50, fraud_probability__lt=0.75).count()
    low_risk           = FraudAlert.objects.filter(fraud_probability__lt=0.50).count()
    context = {
        "total_customers": total_customers, "total_transactions": total_transactions,
        "total_alerts": total_alerts, "pending_alerts": pending_alerts,
        "confirmed_fraud": confirmed_fraud, "model_accuracy": 87.4,
        "recent_alerts": recent_alerts,
        "alert_type_labels": [a["alert_type"] for a in alert_types],
        "alert_type_counts": [a["count"] for a in alert_types],
        "high_risk": high_risk, "medium_risk": medium_risk, "low_risk": low_risk,
    }
    return render(request, "fraud_app/dashboard.html", context)

@login_required
def transaction_list(request):
    transactions = Transaction.objects.select_related("customer","vending_point").order_by("-purchase_date")[:200]
    return render(request, "fraud_app/transactions.html", {"transactions": transactions})

@login_required
def upload_transactions(request):
    if request.method == "POST" and request.FILES.get("csv_file"):
        reader    = csv.DictReader(io.StringIO(request.FILES["csv_file"].read().decode("utf-8")))
        processed = flagged = 0
        errors    = []
        for i, row in enumerate(reader, 1):
            try:
                customer, _ = Customer.objects.get_or_create(
                    meter_number=row["meter_number"].strip(),
                    defaults={"account_number": f"ACC{row['meter_number'].strip()}", "customer_name": f"Customer {row['meter_number'].strip()}"})
                features = {k: float(row.get(k, 0)) for k in [
                    "avg_monthly_consumption","consumption_variance","token_purchase_freq",
                    "purchased_to_measured_ratio","night_consumption_ratio","tamper_flag_count",
                    "authorized_vending","zone_fraud_rate","consumption_anomaly_score","token_reuse_flag"]}
                features["authorized_vending"] = float(row.get("authorized_vending", 1))
                result = predict(features)
                txn, created = Transaction.objects.get_or_create(
                    token_id=row["token_id"].strip(),
                    defaults={"customer": customer, "purchase_date": timezone.now(),
                              "purchased_units": float(row.get("purchased_units", 0)),
                              "payment_amount": float(row.get("payment_amount", 0)),
                              "fraud_score": result["fraud_probability"], "is_flagged": result["is_flagged"]})
                if result["is_flagged"] and created:
                    FraudAlert.objects.create(transaction=txn, customer=customer,
                        alert_type=result["alert_type"], fraud_probability=result["fraud_probability"],
                        investigation_notes=result["top_reason"])
                    flagged += 1
                processed += 1
            except Exception as e:
                errors.append(f"Row {i}: {e}")
        messages.success(request, f"Processed {processed} transactions. {flagged} flagged.")
        if errors: messages.warning(request, f"{len(errors)} errors: " + "; ".join(errors[:3]))
        return redirect("transaction_list")
    return render(request, "fraud_app/upload.html")

# predict_single moved to bottom of file

@login_required
def alert_list(request):
    status_filter = request.GET.get("status", "")
    alerts = FraudAlert.objects.select_related("customer","transaction")
    if status_filter: alerts = alerts.filter(investigation_status=status_filter)
    return render(request, "fraud_app/alerts.html", {"alerts": alerts.order_by("-created_at"), "status_filter": status_filter})

@login_required
def alert_detail(request, pk):
    alert = get_object_or_404(FraudAlert, pk=pk)
    if request.method == "POST":
        alert.investigation_status = request.POST.get("investigation_status")
        alert.investigation_notes  = request.POST.get("investigation_notes", "")
        alert.investigated_by      = request.user
        alert.save()
        messages.success(request, f"Alert #{pk} updated.")
        return redirect("alert_list")
    history = Transaction.objects.filter(customer=alert.customer).order_by("-purchase_date")[:20]
    return render(request, "fraud_app/alert_detail.html", {"alert": alert, "customer_history": history})

@login_required
def seed_demo_data(request):
    if Customer.objects.exists():
        messages.info(request, "Demo data already exists.")
        return redirect("dashboard")
    np.random.seed(99)
    vp_auth   = VendingPoint.objects.create(dealer_name="Authorized Dealer A", authorization_status="Authorized",   location="Lagos Island")
    vp_unauth = VendingPoint.objects.create(dealer_name="Unknown Dealer X",    authorization_status="Unauthorized", location="Unknown")
    zones = ["Lagos Island","Ikeja","Surulere","Lekki","Apapa"]
    categories = ["Residential","Commercial","Industrial"]
    for i in range(1, 51):
        fraud_customer = (i > 45)
        customer = Customer.objects.create(
            account_number=f"ACC{str(i).zfill(5)}", meter_number=f"MTR{str(i).zfill(6)}",
            customer_name=f"Customer {i}", customer_category=random.choice(categories),
            geographic_zone=random.choice(zones))
        for _ in range(3):
            if fraud_customer:
                feats = {"avg_monthly_consumption": float(np.random.uniform(2,20)),
                    "consumption_variance": float(np.random.uniform(0,2)),
                    "token_purchase_freq": float(np.random.randint(10,18)),
                    "purchased_to_measured_ratio": float(np.random.uniform(3,7)),
                    "night_consumption_ratio": float(np.random.uniform(0.75,0.99)),
                    "tamper_flag_count": float(np.random.randint(4,12)),
                    "authorized_vending": 0.0, "zone_fraud_rate": float(np.random.uniform(0.30,0.55)),
                    "consumption_anomaly_score": float(np.random.uniform(3,6)), "token_reuse_flag": 1.0}
                vp = vp_unauth
            else:
                feats = {"avg_monthly_consumption": float(np.random.uniform(50,250)),
                    "consumption_variance": float(np.random.uniform(5,30)),
                    "token_purchase_freq": float(np.random.randint(1,4)),
                    "purchased_to_measured_ratio": float(np.random.uniform(0.9,1.1)),
                    "night_consumption_ratio": float(np.random.uniform(0.15,0.35)),
                    "tamper_flag_count": 0.0, "authorized_vending": 1.0,
                    "zone_fraud_rate": float(np.random.uniform(0.03,0.08)),
                    "consumption_anomaly_score": float(np.random.uniform(0,0.5)), "token_reuse_flag": 0.0}
                vp = vp_auth
            result  = predict(feats)
            token   = "".join(random.choices(string.digits, k=20))
            txn = Transaction.objects.create(
                customer=customer, vending_point=vp, token_id=token,
                purchase_date=timezone.now()-timedelta(days=random.randint(1,30)),
                purchased_units=float(np.random.uniform(10,500)),
                payment_amount=float(np.random.uniform(500,25000)),
                fraud_score=result["fraud_probability"], is_flagged=result["is_flagged"])
            if result["is_flagged"]:
                FraudAlert.objects.create(transaction=txn, customer=customer,
                    alert_type=result["alert_type"], fraud_probability=result["fraud_probability"],
                    investigation_notes=result["top_reason"])
    messages.success(request, "Demo data created: 50 customers, 150 transactions.")
    return redirect("dashboard")

@login_required
def predict_single(request):
    result = None
    saved_customer = None
    saved_alert = None
    form_data = {}

    if request.method == "POST":
        form_data = request.POST

        # ── get or create customer ──
        meter_number   = request.POST.get("meter_number", "").strip() or "MTR000000"
        account_number = request.POST.get("account_number","").strip() or f"ACC{meter_number}"
        customer_name  = request.POST.get("customer_name","").strip()  or "Unknown Customer"

        customer, _ = Customer.objects.get_or_create(
            meter_number=meter_number,
            defaults={
                "account_number":    account_number,
                "customer_name":     customer_name,
                "customer_category": request.POST.get("customer_category","Residential"),
                "geographic_zone":   request.POST.get("geographic_zone",""),
            }
        )
        saved_customer = customer

        # ── build features ──
        keys = ["avg_monthly_consumption","consumption_variance","token_purchase_freq",
                "purchased_to_measured_ratio","night_consumption_ratio","tamper_flag_count",
                "authorized_vending","zone_fraud_rate","consumption_anomaly_score","token_reuse_flag"]
        features = {k: float(request.POST.get(k, 0) or 0) for k in keys}

        # ── run ML model ──
        result = predict(features)

        # ── save transaction ──
        import random, string as st
        token_id = request.POST.get("token_id","").strip() or "".join(random.choices(st.digits, k=20))
        from django.utils import timezone
        txn, created = Transaction.objects.get_or_create(
            token_id=token_id,
            defaults={
                "customer":        customer,
                "purchase_date":   timezone.now(),
                "purchased_units": float(request.POST.get("purchased_units", 0) or 0),
                "payment_amount":  float(request.POST.get("payment_amount",  0) or 0),
                "fraud_score":     result["fraud_probability"],
                "is_flagged":      result["is_flagged"],
            }
        )

        # ── save fraud alert if flagged ──
        if result["is_flagged"]:
            saved_alert = FraudAlert.objects.create(
                transaction=txn,
                customer=customer,
                alert_type=result["alert_type"],
                fraud_probability=result["fraud_probability"],
                investigation_notes=result["top_reason"],
            )

        messages.success(request, f"Prediction saved for {customer_name} ({meter_number}).")

    return render(request, "fraud_app/predict_single.html", {
        "result":         result,
        "saved_customer": saved_customer,
        "saved_alert":    saved_alert,
        "form_data":      form_data,
    })


@login_required
def predict_history(request):
    """Show all saved single predictions — every transaction ever scored."""
    transactions = Transaction.objects.select_related(
        "customer", "vending_point"
    ).order_by("-created_at")

    # search by meter number or customer name
    search = request.GET.get("search", "").strip()
    if search:
        transactions = transactions.filter(
            customer__meter_number__icontains=search
        ) | transactions.filter(
            customer__customer_name__icontains=search
        )

    return render(request, "fraud_app/predict_history.html", {
        "transactions": transactions,
        "search":       search,
    })
