"""
================================================================================
Project VERA — Email Alerting Utility
================================================================================

Sends escalation/discrepancy/action alert emails to a list of recipients
(e.g., safety team, supervisors) when triggered by VERA agents or UI actions.

Environment variables:
  - SENDER_EMAIL:      Gmail address to send from
  - EMAIL_APP_PASSWORD: Gmail App Password (not your regular password)
  - EMAIL_RECIPIENTS:   Comma-separated list of recipient emails

If any credential is missing, the module logs a warning instead of crashing.

Usage:
    from shared.email_utils import send_alert_email
    send_alert_email("Security Alert", "Details here...")
================================================================================
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Email credentials from environment
# ---------------------------------------------------------------------------
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "")

# Recipient list: comma-separated in .env
_raw_recipients = os.getenv("EMAIL_RECIPIENTS", "") or os.getenv("SUPERVISOR_EMAIL", "")
EMAIL_RECIPIENTS = [
    r.strip() for r in _raw_recipients.split(",") if r.strip()
]

# SMTP settings (Gmail)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587


def is_email_configured() -> bool:
    """Check whether email alerting credentials are fully configured."""
    return bool(SENDER_EMAIL and EMAIL_APP_PASSWORD and EMAIL_RECIPIENTS)


def send_alert_email(
    subject: str,
    body: str,
    to_emails: list[str] | None = None,
) -> bool:
    """
    Send an alert email to one or more recipients.

    Args:
        subject: Email subject line.
        body: Email body text (plain text).
        to_emails: Override recipient list. Defaults to EMAIL_RECIPIENTS from .env.

    Returns:
        True if email was sent successfully, False otherwise.
    """
    recipients = to_emails or EMAIL_RECIPIENTS

    if not SENDER_EMAIL or not EMAIL_APP_PASSWORD:
        print(
            "[EMAIL] ⚠️  Email credentials not configured. "
            "Set SENDER_EMAIL and EMAIL_APP_PASSWORD in .env to enable alerts."
        )
        return False

    if not recipients:
        print(
            "[EMAIL] ⚠️  No recipients configured. "
            "Set EMAIL_RECIPIENTS in .env (comma-separated)."
        )
        return False

    # Build the email message
    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = f"[VERA] {subject}"

    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(SENDER_EMAIL, EMAIL_APP_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipients, msg.as_string())

        print(f"[EMAIL] ✅ Alert sent to {', '.join(recipients)}")
        return True

    except smtplib.SMTPAuthenticationError:
        print(
            "[EMAIL] ❌ Authentication failed. "
            "Check SENDER_EMAIL and EMAIL_APP_PASSWORD in .env. "
            "Ensure you are using a Gmail App Password."
        )
        return False

    except Exception as e:
        print(f"[EMAIL] ❌ Failed to send email: {e}")
        return False


# Backward-compatible alias
send_escalation_email = send_alert_email
