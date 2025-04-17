# Strava Activity Analysis

A web application for analyzing and visualizing Strava activity data. This tool provides insights into your training performance, activity patterns, and progress over time.

## Features

- Interactive performance charts with rolling averages
- Activity type breakdown and visualization
- Performance period detection
- Calendar view of activities
- Progress tracking towards goals
- Custom performance scoring system

## Setup

1. Clone the repository:
```bash
git clone https://github.com/zappingseb/stravanalytics.git
cd stravanalytics
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with your Strava API credentials:
```
STRAVA_CLIENT_ID=your_client_id
STRAVA_CLIENT_SECRET=your_client_secret
STRAVA_REFRESH_TOKEN=your_refresh_token
FLASK_SECRET_KEY=your_secret_key
```

4. Run the application:
```bash
python app.py
```

## Usage

1. Access the application at `http://localhost:5000`
2. Select date range and activity types
3. View your activity data and performance metrics

## Security Note

Never commit your `.env` file or expose your Strava API credentials. The `.env` file is included in `.gitignore` for this reason.

## License

MIT License - feel free to use and modify for your own purposes. 