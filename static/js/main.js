// Placeholder for future JS enhancements (e.g. live chatbot interactions).
document.addEventListener("DOMContentLoaded", function () {
    // Auto-dismiss alerts after 5 seconds
    setTimeout(function () {
        document.querySelectorAll(".alert").forEach(function (alert) {
            const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
            if (bsAlert) bsAlert.close();
        });
    }, 5000);
});
