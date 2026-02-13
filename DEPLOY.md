# Deploying to Render

## Quick Setup

### 1. Prepare Your Repository
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-github-repo-url>
git push -u origin main
```

### 2. Deploy on Render

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Render will auto-detect the `render.yaml` configuration

### 3. Set Environment Variables

In Render dashboard, add these environment variables:

```
OPENAI_API_KEY=sk-your-actual-key-here
PINECONE_API_KEY=pcsk_your-actual-key-here
```

### 4. Deploy

Click "Create Web Service" and Render will:
- Use Python 3.11.9 (specified in `runtime.txt`)
- Install dependencies with pre-built wheels
- Start your Streamlit app

## Troubleshooting

### Python Version Issues
- The `runtime.txt` file specifies Python 3.11.9
- This ensures pre-built wheels are available for all dependencies
- Don't use Python 3.14 - it's too new and lacks binary packages

### Build Failures
If you still get build errors:

1. **Check Python version**: Ensure `runtime.txt` contains `python-3.11.9`

2. **Clear build cache**: In Render dashboard → Settings → Clear build cache

3. **Manual build command**: Update `render.yaml` buildCommand to:
   ```yaml
   buildCommand: pip install --upgrade pip setuptools wheel && pip install --only-binary=:all: -r requirements.txt || pip install -r requirements.txt
   ```

### Memory Issues
If the app runs out of memory:
- Upgrade to a paid Render plan (free tier has 512MB RAM)
- Reduce `CHUNK_SIZE` in your code
- Limit `RETRIEVAL_TOP_K` value

### Pinecone Connection
Make sure your Pinecone index region matches your app:
- In the code: `region='us-east-1'`
- Create your Pinecone index in the same region

## Alternative: Use Docker (Optional)

If you prefer Docker deployment:

1. Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. Deploy as Docker service on Render

## Cost Optimization

- **Free Tier**: Limited to 512MB RAM, may experience slowdowns
- **Starter ($7/mo)**: 1GB RAM, suitable for light usage
- **Standard ($25/mo)**: 2GB RAM, recommended for production

## Support

If issues persist:
1. Check Render logs for specific errors
2. Verify all environment variables are set
3. Test locally first: `streamlit run app.py`
