# Customer Call Transcript Analyzer

A professional-grade Python application that analyzes customer call transcripts using AI-powered summarization and sentiment analysis via the Groq API.

## Features

- 🎯 **AI-Powered Analysis**: Uses Groq's advanced language models for accurate summarization and sentiment detection
- 🌐 **Web Interface**: Clean, responsive web UI built with Bootstrap
- 🔗 **REST API**: Programmatic access for integration with other systems
- 📊 **CSV Export**: Automatic saving and downloadable CSV reports
- 📱 **Mobile-Friendly**: Responsive design that works on all devices
- ⚡ **Fast & Reliable**: Built with FastAPI for high performance
- 🛡️ **Production-Ready**: Comprehensive error handling and logging

## Technology Stack

- **Backend**: FastAPI (Python)
- **AI/ML**: Groq API (Mixtral-8x7B model)
- **Frontend**: HTML5, Bootstrap 5, JavaScript
- **Data**: Pandas, CSV
- **Deployment**: Uvicorn ASGI server

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Valid Groq API key

### Quick Start

1. **Clone/Download the repository**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment** (Optional - API key is already included)
   ```bash
   # Edit .env.example to .env file if needed
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Run the application**
   ```bash
   python app.py
   ```
   
   Or with uvicorn:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Access the application**
   - Web Interface: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## Usage

### Web Interface

1. Navigate to http://localhost:8000
2. Enter or paste a customer call transcript
3. Click "Analyze Transcript"
4. View the AI-generated summary and sentiment analysis
5. Results are automatically saved to `call_analysis.csv`

### API Usage

**Analyze a transcript:**
```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"transcript": "Customer was frustrated about payment issues..."}'
```

**Get all results:**
```bash
curl -X GET "http://localhost:8000/api/results"
```

### Sample Transcripts

The application includes sample transcripts for testing:
- **Negative Sentiment**: Payment failure complaint
- **Positive Sentiment**: Successful issue resolution
- **Neutral Sentiment**: General product inquiry

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web interface |
| POST | `/analyze` | Analyze transcript (JSON API) |
| POST | `/analyze-form` | Analyze transcript (Form submission) |
| GET | `/results` | View all results (Web) |
| GET | `/api/results` | Get all results (JSON API) |
| GET | `/health` | Health check |

## CSV Output Format

The application saves results to `call_analysis.csv` with the following columns:

| Column | Description |
|--------|-------------|
| Timestamp | Analysis date and time |
| Transcript | Original customer call transcript |
| Summary | AI-generated 2-3 sentence summary |
| Sentiment | Customer sentiment (Positive/Neutral/Negative) |

## AI Model Configuration

- **Model**: Mixtral-8x7B-32768 (Groq API)
- **Summarization**: 2-3 sentence summaries with 150 token limit
- **Sentiment**: Three-class classification (Positive/Neutral/Negative)
- **Temperature**: 0.3 for summary, 0.1 for sentiment (for consistency)

## Error Handling

The application includes comprehensive error handling for:
- Invalid input validation
- API connection issues
- Rate limiting
- File system errors
- Malformed requests

## Logging

All activities are logged to:
- `app.log` file
- Console output
- Includes timestamps, levels, and detailed error information

## Security Features

- Input validation and sanitization
- XSS protection
- CSRF protection via form tokens
- Rate limiting ready (can be easily added)
- Environment variable protection

## Production Deployment

### Docker (Recommended)

Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables for Production

```bash
GROQ_API_KEY=your_production_api_key
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=False
LOG_LEVEL=WARNING
```

### Reverse Proxy Configuration (Nginx)

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Development

### Running in Development Mode

```bash
# With auto-reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# With debug logging
python app.py
```

### Testing

The application includes comprehensive error handling and validation. For testing:

1. Test with various transcript lengths
2. Test with special characters and formatting
3. Test API endpoints with curl or Postman
4. Verify CSV output format and data integrity

## Troubleshooting

### Common Issues

1. **API Key Issues**
   - Verify your Groq API key is valid
   - Check the `.env` file configuration
   - Ensure no rate limits are exceeded

2. **Import Errors**
   - Run `pip install -r requirements.txt`
   - Ensure Python 3.8+ is being used
   - Check virtual environment activation

3. **Port Already in Use**
   - Change the port in app.py or use: `uvicorn app:app --port 8001`

4. **CSV File Issues**
   - Check write permissions in the application directory
   - Verify no other processes are using the CSV file

### Performance Optimization

- Use Redis for caching API responses
- Implement connection pooling for database storage
- Add CDN for static assets
- Use Gunicorn with multiple workers for production

## Contributing

This is a production-ready application following Python best practices:
- Type hints throughout
- Comprehensive error handling
- Detailed logging
- Clean code structure
- API documentation
- Security considerations

## License

This project is built for educational and commercial purposes. The code follows industry standards and best practices suitable for production deployment.

## Support

For issues or questions:
1. Check the error logs in `app.log`
2. Verify API key and network connectivity
3. Review the troubleshooting section
4. Check Groq API status and rate limits

---

**Note**: This application is production-ready with enterprise-grade features including comprehensive error handling, logging, security measures, and scalable architecture.