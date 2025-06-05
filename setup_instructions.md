# Gold Trading Bot Setup Instructions

## Prerequisites
- Ubuntu server on Digital Ocean
- Python 3.8 or higher

## Installation Steps

1. Connect to your Digital Ocean server via SSH:
   ```
   ssh root@your_server_ip
   ```

2. Update the package list and install dependencies:
   ```
   apt update
   apt upgrade -y
   apt install -y python3-pip python3-dev build-essential ta-lib
   ```

3. Install TA-Lib (Technical Analysis Library):
   ```
   wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
   tar -xzf ta-lib-0.4.0-src.tar.gz
   cd ta-lib/
   ./configure --prefix=/usr
   make
   make install
   cd ..
   rm -rf ta-lib-0.4.0-src.tar.gz ta-lib/
   ```

4. Create a directory for the bot:
   ```
   mkdir -p /opt/gold_trading_bot
   cd /opt/gold_trading_bot
   ```

5. Upload the bot files to the server using SCP:
   ```
   # Run this on your local machine
   scp -r gold_trading_bot.py .env requirements.txt root@your_server_ip:/opt/gold_trading_bot/
   ```

6. Install Python requirements:
   ```
   pip3 install -r requirements.txt
   ```

7. Edit the .env file with your API keys and Telegram details:
   ```
   nano .env
   ```

8. Test the bot:
   ```
   python3 gold_trading_bot.py
   ```

9. Setup as a systemd service:
   ```
   nano /etc/systemd/system/gold-bot.service
   ```

   Paste the following:
   ```
   [Unit]
   Description=Gold Trading Bot
   After=network.target

   [Service]
   User=root
   WorkingDirectory=/opt/gold_trading_bot
   ExecStart=/usr/bin/python3 /opt/gold_trading_bot/gold_trading_bot.py
   Restart=always
   RestartSec=5
   StandardOutput=syslog
   StandardError=syslog
   SyslogIdentifier=goldbot

   [Install]
   WantedBy=multi-user.target
   ```

10. Enable and start the service:
    ```
    systemctl daemon-reload
    systemctl enable gold-bot.service
    systemctl start gold-bot.service
    ```

11. Check the status:
    ```
    systemctl status gold-bot.service
    ```

12. View logs:
    ```
    journalctl -u gold-bot.service -f
    ```

Remember to regularly backup your model files and monitor the bot's performance.
