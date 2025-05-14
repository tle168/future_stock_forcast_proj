import smtplib, ssl
from email.mime.text import MIMEText

# def send_email(data_dict, target_email):
#     content = "\n\n".join([
#         f"{code}: Giá đóng cửa gần nhất = {df['Close'].iloc[-1]:,.2f}"
#         for code, df in data_dict.items() if not df.empty
#     ])
#     msg = MIMEText(content)
#     msg['Subject'] = '📈 Cảnh báo cổ phiếu - Stock Alert'
#     msg['From'] = "python.apps.dev1982@gmail.com"
#     # msg['To'] = "thach.le168@gmail.com"
#     msg['To'] = target_email
#     #msg['Content-Type'] = 'text/html; charset="utf-8"'
    

#     with smtplib.SMTP_SSL('smtp.gmail.com', 587) as smtp:
#         smtp.login("python.apps.dev1982@gmail.com", "trynzifksezihbvp")
#         smtp.send_message(msg)

# from email.message import EmailMessage


# def send_email(ticker, df, target_email):
    
#     host = "smtp.gmail.com"
#     port = 465

#     username = "python.apps.dev1982@gmail.com"
#     password = "trynzifksezihbvp"

#     receiver = target_email
#     context = ssl.create_default_context()

#     email_message = EmailMessage()
#     email_message["Subject"] = "Dự đoán giá cổ phiếu!"
#     email_message.set_content(f"Giá cổ phiếu {ticker} trong những ngày tới!")

#     content= """\
#         <html>
#           <head></head>
#           <body>
#             {0}
#           </body>
#         </html>
#     """.format(df.to_html(index=False))

#     part1 = MIMEText(content, 'html')
#     c.attach(part1)

#     email_message.add_attachment(content, maintype="text", subtype="html")

#     gmail = smtplib.SMTP("smtp.gmail.com", 587)
#     gmail.ehlo()
#     gmail.starttls()
#     gmail.login(username, password)
#     gmail.sendmail(username, receiver, email_message.as_string())
#     gmail.quit()
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_email(df, recipients):
    """
    Sends a Pandas DataFrame as an HTML table in an email.

    Args:
        df (pd.DataFrame): The DataFrame to send.
        subject (str): The email subject.
        recipients (list): A list of recipient email addresses.
        sender_email (str): The sender's email address.
        sender_password (str): The sender's email password.
    """
    subject="Dự đoán giá cổ phiếu!"
    sender_email="python.apps.dev1982@gmail.com"
    sender_password="trynzifksezihbvp"

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = ', '.join(recipients)

    html = f"""
    <html>
    <head></head>
    <body>
        {df.to_html(index=False)}
    </body>
    </html>
    """

    part2 = MIMEText(html, 'html')
    msg.attach(part2)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)  # Use appropriate SMTP server
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipients, msg.as_string())
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")