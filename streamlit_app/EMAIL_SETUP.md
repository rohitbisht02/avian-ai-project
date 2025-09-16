# Avian AI Email OTP Login Setup

## How to set up Gmail for OTP delivery

1. Go to https://myaccount.google.com/apppasswords (You must have 2-Step Verification enabled on your Google account.)
2. Select 'Mail' as the app and 'Other' as the device, name it 'AvianAI', and generate an app password.
3. Copy the generated 16-character app password.
4. Edit `streamlit_app/email_config.yaml` and fill in:

EMAIL_SENDER: "your_email@gmail.com"
EMAIL_PASSWORD: "your_app_password_here"

- Never use your main Gmail password here.
- Keep this file secure and do not share it.

## Example `email_config.yaml`:

EMAIL_SENDER: "your_email@gmail.com"
EMAIL_PASSWORD: "abcd efgh ijkl mnop"

---

If you use a different email provider, update the SMTP logic in `app.py` accordingly.
