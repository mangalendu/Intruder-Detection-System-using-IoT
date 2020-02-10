import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# Email you want to send the update from (only works with gmail)
fromEmail = 'XXXXXXXX'
fromEmailPassword = 'XXXXXXXX'

def sendEmail(image,currentTime, toEmail):
    msg = MIMEMultipart() 
 
    msg['From'] = fromEmail 
 
    msg['To'] = toEmail
    
    msg['Subject'] = "Intruder Detected"

    body = "E-Hawk has detected an intruder! at Time: "+currentTime

    msg.attach(MIMEText(body, 'plain')) 
 
    msg.attach(MIMEImage(image))
    
    smtp = smtplib.SMTP('smtp.gmail.com', 587)
    smtp.starttls()
    smtp.login(fromEmail, fromEmailPassword)
    smtp.sendmail(fromEmail, toEmail, msg.as_string())
    smtp.quit()
