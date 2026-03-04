COMPOSE := $(shell command -v podman >/dev/null 2>&1 && echo "podman compose" || echo "docker compose")

.PHONY: run stop clean

run:
	$(COMPOSE) up --build -d
	@echo ""
	@echo "🔥 TorchCode is running!"
	@echo "   Open http://localhost:8888"
	@echo ""

stop:
	$(COMPOSE) down

clean:
	$(COMPOSE) down -v
	rm -f data/progress.json
