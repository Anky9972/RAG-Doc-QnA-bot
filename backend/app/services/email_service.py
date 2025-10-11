import os
import logging
from typing import Optional
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from jinja2 import Template
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

class EmailService:
    """Service for sending emails"""
    
    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("FROM_EMAIL", self.smtp_username)
        self.from_name = os.getenv("FROM_NAME", "PDF Q&A System")
        
        # Frontend URL for reset links
        self.frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate email configuration"""
        if not self.smtp_username or not self.smtp_password:
            logger.warning("Email service not configured. Set SMTP_USERNAME and SMTP_PASSWORD environment variables.")
            self.enabled = False
        else:
            self.enabled = True
            logger.info(f"Email service configured: {self.smtp_username}")
    
    def is_enabled(self) -> bool:
        """Check if email service is enabled"""
        return self.enabled
    
    def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: Optional[str] = None
    ) -> bool:
        """
        Send an email
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            html_content: HTML content of the email
            text_content: Plain text content (fallback)
        
        Returns:
            bool: True if email sent successfully, False otherwise
        """
        if not self.enabled:
            logger.error("Email service is not enabled")
            return False
        
        try:
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = f"{self.from_name} <{self.from_email}>"
            message["To"] = to_email
            
            # Add text and HTML parts
            if text_content:
                text_part = MIMEText(text_content, "plain")
                message.attach(text_part)
            
            html_part = MIMEText(html_content, "html")
            message.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(message)
            
            logger.info(f"Email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False
    
    def send_password_reset_email(
        self,
        to_email: str,
        username: str,
        reset_token: str,
        expires_in_minutes: int = 60
    ) -> bool:
        """
        Send password reset email
        
        Args:
            to_email: User's email address
            username: User's username
            reset_token: Password reset token
            expires_in_minutes: Token expiry time in minutes
        
        Returns:
            bool: True if email sent successfully
        """
        reset_link = f"{self.frontend_url}/reset-password?token={reset_token}"
        
        # HTML template
        html_template = Template("""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background-color: #4CAF50;
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 5px 5px 0 0;
        }
        .content {
            background-color: #f9f9f9;
            padding: 30px;
            border-radius: 0 0 5px 5px;
        }
        .button {
            display: inline-block;
            padding: 12px 30px;
            background-color: #4CAF50;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            margin: 20px 0;
        }
        .footer {
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-top: 20px;
        }
        .warning {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 12px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Password Reset Request</h1>
        </div>
        <div class="content">
            <p>Hello <strong>{{ username }}</strong>,</p>
            
            <p>We received a request to reset your password for your PDF Q&A System account.</p>
            
            <p>Click the button below to reset your password:</p>
            
            <a href="{{ reset_link }}" class="button">Reset Password</a>
            
            <p>Or copy and paste this link into your browser:</p>
            <p style="word-break: break-all; color: #666;">{{ reset_link }}</p>
            
            <div class="warning">
                <strong>⚠️ Important:</strong> This link will expire in {{ expires_in_minutes }} minutes.
            </div>
            
            <p>If you didn't request a password reset, you can safely ignore this email. Your password will remain unchanged.</p>
            
            <p>For security reasons, never share this link with anyone.</p>
            
            <p>Best regards,<br>
            The PDF Q&A System Team</p>
        </div>
        <div class="footer">
            <p>This is an automated email. Please do not reply to this message.</p>
            <p>&copy; {{ year }} PDF Q&A System. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
        """)
        
        # Plain text template
        text_template = Template("""
Password Reset Request

Hello {{ username }},

We received a request to reset your password for your PDF Q&A System account.

To reset your password, visit this link:
{{ reset_link }}

This link will expire in {{ expires_in_minutes }} minutes.

If you didn't request a password reset, you can safely ignore this email. Your password will remain unchanged.

Best regards,
The PDF Q&A System Team

---
This is an automated email. Please do not reply to this message.
© {{ year }} PDF Q&A System. All rights reserved.
        """)
        
        # Render templates
        html_content = html_template.render(
            username=username,
            reset_link=reset_link,
            expires_in_minutes=expires_in_minutes,
            year=datetime.now().year
        )
        
        text_content = text_template.render(
            username=username,
            reset_link=reset_link,
            expires_in_minutes=expires_in_minutes,
            year=datetime.now().year
        )
        
        # Send email
        return self.send_email(
            to_email=to_email,
            subject="Reset Your Password - PDF Q&A System",
            html_content=html_content,
            text_content=text_content
        )
    
    def send_password_changed_confirmation(
        self,
        to_email: str,
        username: str
    ) -> bool:
        """
        Send password changed confirmation email
        
        Args:
            to_email: User's email address
            username: User's username
        
        Returns:
            bool: True if email sent successfully
        """
        html_template = Template("""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background-color: #4CAF50;
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 5px 5px 0 0;
        }
        .content {
            background-color: #f9f9f9;
            padding: 30px;
            border-radius: 0 0 5px 5px;
        }
        .footer {
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-top: 20px;
        }
        .success {
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 12px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>✓ Password Changed Successfully</h1>
        </div>
        <div class="content">
            <p>Hello <strong>{{ username }}</strong>,</p>
            
            <div class="success">
                Your password has been changed successfully.
            </div>
            
            <p>If you made this change, no further action is required.</p>
            
            <p>If you did NOT change your password, please contact our support team immediately as your account may have been compromised.</p>
            
            <p>Best regards,<br>
            The PDF Q&A System Team</p>
        </div>
        <div class="footer">
            <p>This is an automated email. Please do not reply to this message.</p>
            <p>&copy; {{ year }} PDF Q&A System. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
        """)
        
        html_content = html_template.render(
            username=username,
            year=datetime.now().year
        )
        
        text_content = f"""
Password Changed Successfully

Hello {username},

Your password has been changed successfully.

If you made this change, no further action is required.

If you did NOT change your password, please contact our support team immediately.

Best regards,
The PDF Q&A System Team
        """
        
        return self.send_email(
            to_email=to_email,
            subject="Password Changed - PDF Q&A System",
            html_content=html_content,
            text_content=text_content
        )


# Global instance
email_service = EmailService()