.PHONY: deploy update restart status logs upgrade

upgrade: update deploy restart

define SERVICE_FILE
[Unit]
Description=Nanobot gateway service
After=network-online.target
Wants=network-online.target

[Service]
User=om
Group=om
WorkingDirectory=/home/om/nanobot
EnvironmentFile=/home/om/.nanobot/workspace/.env
Environment="PATH=/home/om/nanobot/.venv/bin:/home/om/.local/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/om/nanobot/.venv/bin/nanobot gateway
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
endef
export SERVICE_FILE

update:
	git pull origin development
	uv pip install -e .
	ln -sf /home/om/nanobot/.venv/bin/nanobot /home/om/.local/bin/nanobot

deploy:
	echo "$$SERVICE_FILE" | sudo tee /etc/systemd/system/nanobot-gateway.service > /dev/null
	sudo systemctl daemon-reload
	sudo systemctl enable nanobot-gateway
	ln -sf /home/om/nanobot/.venv/bin/nanobot /home/om/.local/bin/nanobot


restart:
	sudo systemctl restart nanobot-gateway

status:
	sudo systemctl status nanobot-gateway

logs:
	sudo journalctl -u nanobot-gateway -f
