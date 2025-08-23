# Deploying TTS Service to Render

This guide will walk you through deploying your Gurukul TTS Service to Render.com.

## Prerequisites

1. A [Render.com](https://render.com) account (free tier available)
2. Your code in a Git repository (GitHub, GitLab, or Bitbucket)
3. The deployment files created in this directory

## Deployment Files Created

The following files have been created for your deployment:

- `Dockerfile` - Container configuration for the TTS service
- `render.yaml` - Render deployment configuration
- `requirements_minimal.txt` - Updated with production dependencies
- `start.sh` - Startup script for proper initialization
- `DEPLOY_TO_RENDER.md` - This documentation file

## Step-by-Step Deployment Instructions

### 1. Push Your Code to Git Repository

First, ensure all your code (including the new deployment files) is committed to a Git repository:

```bash
git add .
git commit -m "Add Render deployment configuration"
git push origin main
```

### 2. Connect to Render

1. Go to [Render.com](https://render.com) and sign up/log in
2. Click "New +" and select "Web Service"
3. Connect your Git repository containing the TTS service

### 3. Configure the Web Service

When setting up the service in Render:

**Basic Configuration:**
- **Name**: `gurukul-tts-service` (or your preferred name)
- **Region**: Choose closest to your users (Oregon, Singapore, Frankfurt, etc.)
- **Branch**: `main` (or your default branch)
- **Runtime**: Docker

**Build & Deploy:**
- **Docker Context**: `.` (current directory)
- **Dockerfile Path**: `./Dockerfile`

**Environment Variables:**
Add these environment variables in the Render dashboard:
- `ENVIRONMENT`: `production`
- `PORT`: `8007` (Render will override this automatically)

### 4. Advanced Configuration (Optional)

If you prefer using the render.yaml file for configuration:

1. In your Render dashboard, go to "Account Settings"
2. Connect your repository for Infrastructure as Code
3. Render will automatically detect and use the `render.yaml` file

### 5. Deploy and Monitor

1. Click "Create Web Service"
2. Render will automatically:
   - Build your Docker container
   - Install dependencies
   - Start the TTS service
   - Provide you with a public URL

## Service Endpoints

Once deployed, your TTS service will be available at:
- Base URL: `https://your-service-name.onrender.com`

### Available API Endpoints:

- `GET /` - Service information and health check
- `POST /api/generate` - Generate TTS audio from text
- `GET /api/audio/{filename}` - Retrieve generated audio files
- `GET /api/list-audio-files` - List all available audio files
- `POST /api/generate/stream` - Stream TTS audio directly
- `GET /api/health` - Detailed health check

## Testing Your Deployment

### 1. Health Check
```bash
curl https://your-service-name.onrender.com/api/health
```

### 2. Generate TTS Audio
```bash
curl -X POST https://your-service-name.onrender.com/api/generate \\
     -H "Content-Type: application/x-www-form-urlencoded" \\
     -d "text=Hello, this is a test of the TTS service"
```

### 3. Test with HTML Client
Upload `simple_tts_test.html` to a web server and update the API URL to point to your Render deployment.

## Important Notes

### Free Tier Limitations
- Render free tier puts services to sleep after 15 minutes of inactivity
- Cold start time: 30-60 seconds when service wakes up
- 750 hours/month limit (about 31 days)

### Production Considerations

1. **Upgrade to Paid Plan**: For production use, consider upgrading to avoid sleep/cold starts
2. **Storage**: Audio files are stored in `/app/tts_outputs` with 1GB disk space
3. **CORS**: Update CORS settings in `tts.py` with your actual frontend domains
4. **Monitoring**: Use Render's built-in monitoring or integrate external monitoring

### Environment Variables

Set these in Render dashboard under Environment:
- `ENVIRONMENT=production`
- `PORT` (automatically set by Render)

## Troubleshooting

### Common Issues:

1. **Build Failures**: Check the build logs in Render dashboard
2. **TTS Engine Issues**: The service includes espeak and festival for TTS functionality
3. **Audio Generation Fails**: Check if the container has proper audio system access

### Logs Access:
- View real-time logs in the Render dashboard
- Use the "Logs" tab in your service page

### Debug Commands:
```bash
# Test TTS engine in container
python3 -c "import pyttsx3; engine = pyttsx3.init(); print('TTS OK')"

# Check audio output directory
ls -la /app/tts_outputs/
```

## Scaling and Performance

### Auto-scaling:
The current configuration allows:
- Min instances: 1
- Max instances: 3
- Automatic scaling based on load

### Performance Tips:
1. Use the streaming endpoint (`/api/generate/stream`) for better performance
2. Implement caching for frequently requested text
3. Consider audio compression for better transfer speeds

## Security Considerations

1. **API Rate Limiting**: Implement rate limiting for production use
2. **Input Validation**: The service limits text to 10,000 characters
3. **CORS Configuration**: Update allowed origins for production
4. **Audio File Cleanup**: Implement cleanup for old audio files

## Support

If you encounter issues:
1. Check Render's build and runtime logs
2. Verify all dependencies in `requirements_minimal.txt`
3. Test locally with Docker before deploying
4. Check Render's status page for platform issues

## Local Testing with Docker

Before deploying, test locally:

```bash
# Build the Docker image
docker build -t tts-service .

# Run the container
docker run -p 8007:8007 tts-service
```

Then test at `http://localhost:8007`

---

Your TTS service should now be successfully deployed to Render! 🚀