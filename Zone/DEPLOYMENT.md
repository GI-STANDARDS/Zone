# Deployment Guide

## 🚀 Production Deployment

This guide covers deploying the Face Recognition System in production environments.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.10 or higher
- **RAM**: 8GB+ recommended
- **Storage**: 10GB+ for models and data
- **GPU**: Optional, CUDA-compatible for acceleration

### Dependencies
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install python3-dev python3-pip python3-venv

# Install system dependencies (CentOS/RHEL)
sudo yum install python3-devel python3-pip
```

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd "Face Reco"
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

## ⚙️ Configuration

### Environment Variables
Create `.env` file with following variables:

```bash
# Server Configuration
FACE_RECO_HOST=0.0.0.0
FACE_RECO_PORT=7860
FACE_RECO_SHARE=false
FACE_RECO_DEBUG=false

# Model Configuration
FACE_RECO_DETECTOR=mtcnn
FACE_RECO_ENCODER=facenet
FACE_RECO_DATABASE=sqlite
FACE_RECO_THRESHOLD=0.6

# Performance Configuration
FACE_RECO_BATCH_SIZE=32
FACE_RECO_NUM_WORKERS=4
FACE_RECO_DEVICE=auto
FACE_RECO_CACHE_MODELS=true

# Security Configuration
FACE_RECO_MAX_FILE_SIZE=10485760
FACE_RECO_ENABLE_ETHICS=true
```

### Configuration Files
You can also use YAML or JSON configuration files:

#### YAML Configuration (`config.yaml`)
```yaml
server:
  host: "0.0.0.0"
  port: 7860
  share: false
  debug: false

models:
  detector: "mtcnn"
  encoder: "facenet"
  database: "sqlite"
  threshold: 0.6

performance:
  batch_size: 32
  num_workers: 4
  device: "auto"
  cache_models: true

security:
  max_file_size: 10485760
  enable_ethics_warning: true
```

#### JSON Configuration (`config.json`)
```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 7860,
    "share": false,
    "debug": false
  },
  "models": {
    "detector": "mtcnn",
    "encoder": "facenet",
    "database": "sqlite",
    "threshold": 0.6
  },
  "performance": {
    "batch_size": 32,
    "num_workers": 4,
    "device": "auto",
    "cache_models": true
  },
  "security": {
    "max_file_size": 10485760,
    "enable_ethics_warning": true
  }
}
```

## 🐳 Docker Deployment

### 1. Create Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/embeddings data/images data/database logs

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "src/main.py", "--host", "0.0.0.0", "--port", "7860"]
```

### 2. Build and Run
```bash
# Build image
docker build -t face-recognition .

# Run container
docker run -p 7860:7860 -v $(pwd)/data:/app/data face-recognition
```

### 3. Docker Compose
Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  face-recognition:
    build: .
    ports:
      - "7860:7860"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - FACE_RECO_HOST=0.0.0.0
      - FACE_RECO_PORT=7860
      - FACE_RECO_DEBUG=false
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

## 🌐 Web Server Deployment

### Nginx Reverse Proxy
Create Nginx configuration `/etc/nginx/sites-available/face-recognition`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support for Gradio
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # File upload size limit
    client_max_body_size 10M;
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/face-recognition /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Apache Reverse Proxy
Create Apache configuration `/etc/apache2/sites-available/face-recognition.conf`:

```apache
<VirtualHost *:80>
    ServerName your-domain.com
    ProxyPreserveHost On
    ProxyRequests Off
    
    ProxyPass / http://127.0.0.1:7860/
    ProxyPassReverse / http://127.0.0.1:7860/
    
    # WebSocket support
    ProxyPass /ws ws://127.0.0.1:7860/ws
    ProxyPassReverse /ws ws://127.0.0.1:7860/ws
    
    # File upload size
    LimitRequestBody 10485760
</VirtualHost>
```

Enable site:
```bash
sudo a2ensite face-recognition
sudo apache2ctl configtest
sudo systemctl reload apache2
```

## 🔒 Security Considerations

### 1. Firewall Configuration
```bash
# Allow only necessary ports
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

### 2. SSL/TLS Setup
Using Let's Encrypt with Nginx:
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### 3. Application Security
- Use environment variables for sensitive data
- Enable HTTPS in production
- Regularly update dependencies
- Monitor logs for suspicious activity
- Implement rate limiting
- Use reverse proxy for additional security

## 📊 Monitoring

### 1. Application Monitoring
```bash
# Run with monitoring
python src/main.py --debug --port 7860 > logs/app.log 2>&1 &
```

### 2. System Monitoring
Create monitoring script `monitor.sh`:
```bash
#!/bin/bash
# Check if application is running
if ! pgrep -f "python src/main.py" > /dev/null; then
    echo "$(date): Face recognition service is down" >> logs/monitor.log
    # Restart service
    systemctl restart face-recognition
fi

# Check disk space
df -h /path/to/app >> logs/disk.log
```

### 3. Log Rotation
Create logrotate configuration `/etc/logrotate.d/face-recognition`:
```
/path/to/app/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 user user
    postrotate
        systemctl reload face-recognition
    endscript
}
```

## 🚀 Production Checklist

### Pre-Deployment
- [ ] Test in staging environment
- [ ] Verify all dependencies
- [ ] Check configuration files
- [ ] Test database operations
- [ ] Validate model loading
- [ ] Test file upload functionality

### Deployment
- [ ] Set up production server
- [ ] Configure firewall rules
- [ ] Set up SSL certificates
- [ ] Configure reverse proxy
- [ ] Deploy application files
- [ ] Set up monitoring
- [ ] Configure log rotation

### Post-Deployment
- [ ] Verify application accessibility
- [ ] Test all features
- [ ] Monitor performance
- [ ] Check error logs
- [ ] Set up backup procedures
- [ ] Document deployment

## 🔧 Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Find process using port
sudo lsof -i :7860

# Kill process
sudo kill -9 <PID>
```

#### 2. Permission Denied
```bash
# Fix file permissions
sudo chown -R user:user /path/to/app
chmod +x src/main.py
```

#### 3. Model Loading Issues
```bash
# Check model files
ls -la models/

# Download models manually if needed
python -c "from facenet_pytorch import MTCNN; print('Models loaded successfully')"
```

#### 4. Database Issues
```bash
# Check database permissions
ls -la data/database/

# Test database connection
python -c "from src.database import create_database; db = create_database(); print('Database OK')"
```

### Performance Optimization

#### 1. GPU Acceleration
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install GPU packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Memory Optimization
```bash
# Monitor memory usage
python src/main.py --debug &
top -p $(pgrep -f "python src/main.py")
```

#### 3. Database Optimization
- Use FAISS for large datasets
- Implement database indexing
- Regular database maintenance
- Consider database replication

## 📞 Support

### Log Locations
- Application logs: `logs/face_recognition.log`
- Error logs: `logs/error.log`
- Access logs: Web server logs

### Health Checks
```bash
# Check application status
curl http://localhost:7860

# Check system resources
df -h
free -h
```

### Backup Procedures
```bash
# Backup database
cp data/database/faces.db backups/faces_$(date +%Y%m%d).db

# Backup configuration
cp config.yaml backups/config_$(date +%Y%m%d).yaml

# Backup models
tar -czf backups/models_$(date +%Y%m%d).tar.gz models/
```

---

## 🎯 Production Best Practices

1. **Security First**
   - Always use HTTPS
   - Implement authentication
   - Regular security updates
   - Monitor access logs

2. **Performance Monitoring**
   - Track response times
   - Monitor resource usage
   - Set up alerts
   - Regular performance tuning

3. **Reliability**
   - Implement health checks
   - Set up auto-restart
   - Regular backups
   - Disaster recovery plan

4. **Scalability**
   - Load balancing for high traffic
   - Database optimization
   - CDN for static assets
   - Horizontal scaling plan

5. **Maintenance**
   - Regular updates
   - Log rotation
   - Database maintenance
   - Performance tuning
