.PHONY: deploy update restart status logs upgrade

upgrade: update deploy restart

define SERVICE_FILE
[Unit]
Description=Nanobot gateway service
After=network-online.target
Wants=network-online.target

[Service]
WorkingDirectory=/home/om/nanobot
EnvironmentFile=/home/om/.nanobot/workspace/.env
Environment="PATH=/home/om/nanobot/.venv/bin:/home/om/.local/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/om/nanobot/.venv/bin/nanobot gateway
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
endef
export SERVICE_FILE

update:
	git fetch origin
	git reset --hard origin/development
	uv pip install -e .
	ln -sf /home/om/nanobot/.venv/bin/nanobot /home/om/.local/bin/nanobot

deploy:
	mkdir -p ~/.config/systemd/user
	echo "$$SERVICE_FILE" > ~/.config/systemd/user/nanobot-gateway.service
	systemctl --user daemon-reload
	systemctl --user enable nanobot-gateway
	ln -sf /home/om/nanobot/.venv/bin/nanobot /home/om/.local/bin/nanobot

restart:
	systemctl --user restart nanobot-gateway

status:
	systemctl --user status nanobot-gateway

logs:
	journalctl --user -u nanobot-gateway -f
