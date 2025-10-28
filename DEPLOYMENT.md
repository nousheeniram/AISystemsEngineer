# Deployment Guide

## Replit Deployment Configuration

The application is configured for **Autoscale Deployment** on Replit, which is ideal for stateless REST APIs.

### Deployment Settings

**Deployment Type:** Autoscale  
**Run Command:**
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 5000
```

### Why Autoscale?

Autoscale deployment is perfect for this ML API because:
- **Stateless**: The API doesn't maintain state in server memory (model is loaded from disk)
- **Cost-effective**: Only runs when requests are made
- **Auto-scaling**: Handles traffic spikes automatically
- **Simple**: No complex orchestration needed

### Pre-Deployment Checklist

Before deploying to production:

1. **Train the Model**
   ```bash
   python train_pipeline.py
   ```
   Ensure `models/latest_model.pkl` exists and is recent.

2. **Run Tests**
   ```bash
   pytest tests/ -v
   ```
   All tests should pass (9/9).

3. **Verify Health Check**
   ```bash
   curl http://localhost:5000/health
   ```
   Should return: `{"status": "healthy", "model_loaded": true}`

4. **Test Predictions Locally**
   ```bash
   curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"fixed_acidity": 7.4, "volatile_acidity": 0.7, "citric_acid": 0.0, "residual_sugar": 1.9, "chlorides": 0.076, "free_sulfur_dioxide": 11.0, "total_sulfur_dioxide": 34.0, "density": 0.9978, "pH": 3.51, "sulphates": 0.56, "alcohol": 9.4}'
   ```

### Deployment Steps on Replit

1. Click the **Deploy** button in the Replit interface
2. Select **Autoscale Deployment**
3. The configuration is already set (no changes needed)
4. Click **Deploy**
5. Wait for health checks to pass
6. Your API will be live at the provided URL!

### Health Check Endpoint

The deployment uses `/` as the health check endpoint, which returns:

```json
{
  "message": "Wine Quality Prediction API",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "predict": "/predict",
    "docs": "/docs"
  }
}
```

This ensures quick health check responses (< 1 second).

### Environment Variables

No environment variables are required for basic deployment. The configuration is managed through `config/config.yaml`.

### Monitoring & Troubleshooting

**Check Deployment Logs:**
- View logs in the Replit Deployments tab
- Look for "Application startup complete" message

**Common Issues:**

1. **Model Not Found**
   - Error: "Model not loaded. Run training first."
   - Fix: Run `python train_pipeline.py` before deploying

2. **Port Mismatch**
   - Ensure the app binds to port 5000
   - Check `config/config.yaml` → `api.port: 5000`

3. **Slow Health Checks**
   - The root endpoint `/` responds instantly
   - Model loading happens on startup, not per-request

### API Documentation

Once deployed, your API documentation is available at:
- **Swagger UI**: `https://your-deployment-url.repl.co/docs`
- **ReDoc**: `https://your-deployment-url.repl.co/redoc`

### Production Considerations

For production use, consider:

1. **Model Versioning**: Track model versions in MLflow
2. **Performance Monitoring**: Add metrics collection (Prometheus/Grafana)
3. **Rate Limiting**: Protect against abuse
4. **Authentication**: Add API key authentication
5. **Automated Retraining**: Schedule periodic model updates
6. **A/B Testing**: Deploy multiple model versions

### CI/CD Integration

The GitHub Actions workflow (`.github/workflows/ml-pipeline.yml`) automatically:
- Trains the model on push to main
- Runs tests
- Uploads model artifacts

You can extend this to trigger Replit deployments automatically.

---

**Ready to Deploy?** Click the Deploy button and your ML API will be live in minutes!
