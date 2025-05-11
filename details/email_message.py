import smtplib
from email.mime.text import MIMEText


def send(modelname):
    #连接SMTP服务器
    smtp_server = ""
    smtp_port = 25
    smtp_obj = smtplib.SMTP(smtp_server, smtp_port)
    #登录邮箱账号
    email_address = ""
    password = ""

    smtp_obj.login(email_address, password)
    #创建邮件
    msg = MIMEText("")
    msg["From"] = email_address
    msg["To"] = ""
    msg["Subject"] = modelname + "\t finished"
    #发送邮件
    recipient = ""
    smtp_obj.sendmail(email_address, recipient, msg.as_string())
    # 关闭连接
    smtp_obj.quit()

if __name__ == "__main__":
    modelname = 'mult'
    # 调用主函数
    send(modelname)