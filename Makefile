APP_NAME=predictor-app
DOCKERFILE=Dockerfile

.PHONY: build run clean

# Construiește imaginea Docker
build:
	docker build -t $(APP_NAME) -f $(DOCKERFILE) .

# Rulează aplicația cu montare director local (acces la input.csv)
run:
	docker run --rm -v ${PWD}:/app $(APP_NAME)

# Șterge imaginea Docker (curățare)
clean:
	docker rmi $(APP_NAME)
