APP_NAME=predictor-app
DOCKERFILE_PREDICTOR=Dockerfile-predictor
DOCKERFILE_BUILD=Dockerfile-predictor

.PHONY: build run clean

# Construiește imaginea Docker
build:
	docker build -t $(APP_NAME) -f $(DOCKERFILE_BUILD) .
	docker build -t $(APP_NAME) -f $(DOCKERFILE_PREDICTOR) .
	docker run --rm -v ${PWD}:/app $(APP_NAME) python model_predictor.py

# Rulează aplicația cu montare director local (acces la input.csv)
predict:
	docker run --rm -v ${PWD}:/app $(APP_NAME)

# Șterge imaginea Docker (curățare)
clean:
	docker rmi $(APP_NAME)
