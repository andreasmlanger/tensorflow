"""
Sending an email with attached photo via Sendgrid API
"""

from dotenv import load_dotenv
import sendgrid
from sendgrid.helpers.mail import Attachment, Content, ContentId, Disposition, FileContent, FileName, FileType, Mail
import os

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

SENDGRID_API_KEY = os.environ.get('SENDGRID_API_KEY')
FROM_EMAIL = os.environ.get('FROM_EMAIL')
TO_EMAIL = os.environ.get('TO_EMAIL')


def send_email_notification(im, current_time):
    a = image_to_attachment(im)
    txt = f'Intruder detected at {current_time}'
    subject = 'Security alert in your home!'
    send_email(txt, subject, a)  # send email notification


def image_to_attachment(image):
    return Attachment(FileContent(image),
                      FileName('photo.png'),
                      FileType('image/png'),
                      Disposition('inline'),
                      ContentId('Photo'))


def send_email(txt, subject, attachment=None):
    try:
        sg = sendgrid.SendGridAPIClient(api_key=SENDGRID_API_KEY)
        content = Content('text/html', txt)
        mail = Mail(FROM_EMAIL, TO_EMAIL, subject, content)
        mail.attachment = attachment
        mail_json = mail.get()  # JSON-ready representation of the Mail object
        sg.client.mail.send.post(request_body=mail_json)  # send an HTTP POST request to /mail/send
        print(f'Email sent to {TO_EMAIL}')
    except Exception as ex:
        print(ex)
