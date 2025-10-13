# News Topic Classification - Azure Deployment

This project demonstrates how to deploy a **News Topic Classification model** (trained on the AG News dataset) as a Flask web service in **Azure App Service using Docker**. The deployment includes handling model and vectorizer serialization, Docker containerization, and Azure Container Registry (ACR) integration.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Prerequisites](#prerequisites)  
- [Docker Containerization](#docker-containerization)  
- [Azure Deployment](#azure-deployment)  
- [Testing](#testing)  
- [Challenges & Solutions](#challenges--solutions)

---

## Project Overview

This project includes:

- A Flask app exposing a `/predict` endpoint for news text classification.
- Serialized model (`best_model.pkl`) and vectorizer (`best_vectorizer.pkl`) files.
- Dockerfile for containerizing the Flask app.
- Deployment to Azure App Service via Azure Container Registry.

---

## Prerequisites

- Python 3.10+  
- Docker (with BuildKit enabled)  
- Azure CLI  
- Azure subscription (free tier is sufficient)  

---

## Docker Containerization

1. **Create Dockerfile**

2. **Build the Docker image (without cache to avoid architecture or stale dependency issues):**
```bash
export DOCKER_BUILDKIT=1
docker buildx build --platform linux/amd64 --no-cache -t agnewsregistry12345.azurecr.io/ag-news-azure:latest .
```

3. **Run locally to test:**
```bash
docker run -p 8080:8080 ag-news-azure
curl -X POST http://127.0.0.1:8080/predict -H "Content-Type: application/json" -d '{"text": "SpaceX launches new rocket"}'
```

---

## Azure Deployment

1. **Create Resource Group & App Service Plan**
```bash
az group create --name agnews-rg --location "centralus"

az appservice plan create \
  --name agnews-plan \
  --resource-group agnews-rg \
  --sku B1 \
  --is-linux \
  --location "centralus"
```

2. **Create Azure Container Registry (ACR)**
```bash
az acr create --resource-group agnews-rg --name agnewsregistry12345 --sku Basic
az acr update -n agnewsregistry12345 --admin-enabled true
```

3. **Push Docker Image to ACR**
```bash
az acr login --name agnewsregistry12345
docker push agnewsregistry12345.azurecr.io/ag-news-azure:latest
```

4. **Create Web App & Configure Container**
```bash
az webapp create --resource-group agnews-rg --plan agnews-plan --name agnews-webapp --deployment-container-image-name agnewsregistry12345.azurecr.io/ag-news-azure:latest

az webapp config container set \
  --name agnews-webapp \
  --resource-group agnews-rg \
  --container-registry-url agnewsregistry12345.azurecr.io \
  --container-registry-user "$(az acr credential show -n agnewsregistry12345 --query username -o tsv)" \
  --container-registry-password "$(az acr credential show -n agnewsregistry12345 --query "passwords[0].value" -o tsv)" \
  --container-image-name agnewsregistry12345.azurecr.io/ag-news-azure:latest

az webapp restart --name agnews-webapp --resource-group agnews-rg
```

---

## Testing Deployment

Test the live endpoint:
```bash
curl -X POST https://agnews-webapp.azurewebsites.net/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "SpaceX launches new rocket into orbit"}'
```

Example Response:
```json
{
  "prediction": 2
}
```

![Azure Console Prediction](images/AgNewsPredictionAzureWebApp.png)

---

## Challenges & Solutions:

1. **AWS Lambda architecture mismatch** 
	- Problem: exec format error due to arm64 vs amd64 mismatch.
	- Solution: Rebuilt Docker image explicitly for linux/amd64.
2. **Flask JSON serialization error**
	- Problem: TypeError: Object of type int64 is not JSON serializable.
	- Solution: Cast predictions to Python int before returning JSON:
	```bash
	return jsonify({"prediction": int(prediction)})
	```
3. **Azure subscription namespace not registered**
	- Problem: MissingSubscriptionRegistration errors for Microsoft.Web and Microsoft.ContainerRegistry.
	- Solution: Registered namespaces via Azure portal or CLI.
4. **App Service Plan quota issues**
	- Problem: Additional quota required for B1 plan.
	- Solution: Requested higher quota or used available free tier location.
		- tried: eastus, eastus2, centralus (worked)
5. **Azure container fails immediately / Site failing to start**
	- Problem: Container architecture or working directory mismatch.
	- Solution:
		-Rebuilt image with **--platform=linux/amd64**.
		-Corrected **WORKDIR** in Dockerfile.
		-Verified Flask app exposes port **8080**.
