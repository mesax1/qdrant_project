# Instructions to run the project

## Create a virtual environment

```bash
python -m venv venv
```

## Activate the virtual environment

Linux/MacOS:
```bash
source venv/bin/activate
```

Windows:
```bash
.\venv\Scripts\activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run the project

### Kubernetes setup and port forwarding

First, add Qdrant helm chart repository:
```bash
helm repo add qdrant https://qdrant.github.io/helm-charts
```

Navigate to the project directory where the `qdrant-values.yaml` file is located.

Then, install the Qdrant helm chart with the following command:
```bash
helm install qdrant qdrant/qdrant -f qdrant-values.yaml --namespace default
```

Then, port forward the Qdrant service to your local machine:
```bash
kubectl port-forward svc/qdrant-service 6333:6333
```

### Run the project

Run the following command to index the data:
```bash
python index_data.py
```

Then, run the following command to activate the FastAPI server for hybrid search:
```bash
python service.py
```

Access the API docs at:
```bash
http://localhost:8000/docs
```

Another alternative to just run a hybrid search is to use the `hybrid_searcher.py` file to execute a hybrid search example directly.

```bash
python hybrid_searcher.py
```

