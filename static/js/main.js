document.addEventListener("DOMContentLoaded", function () {
    // Auto-dismiss alerts after 5s
    setTimeout(function () {
        document.querySelectorAll(".alert").forEach(function (alert) {
            const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
            if (bsAlert) bsAlert.close();
        });
    }, 5000);
});
