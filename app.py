from flask import Flask, render_template, jsonify, request, redirect, session, url_for
import os
import requests
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
from functools import lru_cache
import time
import numpy as np
from scipy.optimize import curve_fit

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev')  # Add to .env

CLIENT_ID = os.getenv('STRAVA_CLIENT_ID')
CLIENT_SECRET = os.getenv('STRAVA_CLIENT_SECRET')

colors = {
            'Run': '#FC4C02',      # Strava orange
            'RunStroller': '#FF8C69',  # Lighter orange for stroller runs
            'Ride': '#2D87C9',     # Blue
            'Swim': '#1BB6AF',     # Teal
            'Walk': '#7CB342',     # Green
            'Hike': '#8D6E63',     # Brown
            'NordicSki': '#4A90E2' # Light blue
        }

# Cache to store activities with timestamps
activity_cache = {}
CACHE_TTL = 300  # 5 minutes in seconds

def get_cache_key(after, before, activity_types):
    """Generate a unique cache key for the request."""
    if activity_types:
        activity_types = sorted(activity_types)  # Sort to ensure consistent keys
    return f"{after}_{before}_{','.join(activity_types) if activity_types else 'all'}"

def get_cached_activities(after, before, activity_types):
    """Get activities from cache if they exist and haven't expired."""
    cache_key = get_cache_key(after, before, activity_types)
    cached_data = activity_cache.get(cache_key)
    
    if cached_data:
        timestamp, activities = cached_data
        if time.time() - timestamp < CACHE_TTL:
            print("Cache hit!")
            return activities
        else:
            # Remove expired cache entry
            del activity_cache[cache_key]
    
    return None

def cache_activities(after, before, activity_types, activities):
    """Store activities in cache with current timestamp."""
    cache_key = get_cache_key(after, before, activity_types)
    activity_cache[cache_key] = (time.time(), activities)

def clean_expired_cache():
    """Remove expired entries from cache."""
    current_time = time.time()
    expired_keys = [
        key for key, (timestamp, _) in activity_cache.items()
        if current_time - timestamp >= CACHE_TTL
    ]
    for key in expired_keys:
        del activity_cache[key]

def get_token():
    auth_url = "https://www.strava.com/oauth/token"
    payload = {
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'grant_type': 'refresh_token',
        'refresh_token': os.getenv('STRAVA_REFRESH_TOKEN')
    }
    
    try:
        res = requests.post(auth_url, data=payload)
        access_token = res.json()['access_token']
        return access_token
    except Exception as e:
        print(f"Error getting token: {e}")
        return None

def get_activities(access_token, after=None, before=None, activity_types=None):
    # Check cache first
    cached_result = get_cached_activities(after, before, activity_types)
    if cached_result is not None:
        return cached_result
        
    # Clean expired cache entries periodically
    clean_expired_cache()
    
    activities_url = "https://www.strava.com/api/v3/athlete/activities"
    header = {'Authorization': f'Bearer {access_token}'}
    all_activities = []
    
    # Convert date strings to timestamps
    after_ts = int(datetime.strptime(after, '%Y-%m-%d').timestamp()) if after else None
    before_ts = int(datetime.strptime(before, '%Y-%m-%d').timestamp()) if before else None
    
    # If the range is more than 6 months, split into chunks
    if after_ts and before_ts and (before_ts - after_ts) > 15552000:  # 180 days in seconds
        current_after = after_ts
        chunk_size = 15552000  # 180 days in seconds
        
        while current_after < before_ts:
            current_before = min(current_after + chunk_size, before_ts)
            
            # Get activities for this chunk with pagination
            page = 1
            while True:
                params = {
                    'per_page': 200,
                    'page': page,
                    'after': current_after,
                    'before': current_before
                }
                
                try:
                    response = requests.get(activities_url, headers=header, params=params)
                    
                    if not response.ok:
                        print(f"Response not OK: {response.status_code}")
                        break
                        
                    chunk_activities = response.json()
                    
                    if not isinstance(chunk_activities, list):
                        print(f"Activities is not a list but: {type(chunk_activities)}")
                        break
                        
                    if not chunk_activities:  # No more activities in this chunk
                        break
                        
                    # Process activities
                    for activity in chunk_activities:
                        if not activity_types or activity['type'] in activity_types:
                            # Determine if it's a stroller run
                            activity_type = activity['type']
                            if activity_type == 'Run' and 'Ausflug' in activity['name']:
                                activity_type = 'RunStroller'
                                
                            all_activities.append({
                                'date': datetime.strptime(activity['start_date'][:10], '%Y-%m-%d'),
                                'distance': activity['distance'] / 1000,  # Convert to km
                                'name': activity['name'],
                                'type': activity_type,  # Use modified type
                                'moving_time': activity['moving_time'] / 3600,  # Convert to hours
                                'average_speed': (activity['distance'] / 1000) / (activity['moving_time'] / 3600),  # km/h,
                                'total_elevation_gain': activity['total_elevation_gain']
                            })
                    
                    if len(chunk_activities) < 200:  # Last page for this chunk
                        break
                        
                    page += 1
                    
                except Exception as e:
                    print(f"Error getting activities: {e}")
                    break
            
            current_after = current_before
    else:
        # For shorter ranges, just use pagination
        page = 1
        while True:
            params = {
                'per_page': 200,
                'page': page
            }
            
            if after_ts:
                params['after'] = after_ts
            if before_ts:
                params['before'] = before_ts
            
            try:
                response = requests.get(activities_url, headers=header, params=params)
                
                if not response.ok:
                    print(f"Response not OK: {response.status_code}")
                    break
                    
                activities = response.json()
                
                if not isinstance(activities, list):
                    print(f"Activities is not a list but: {type(activities)}")
                    break
                    
                if not activities:  # No more activities
                    break
                    
                # Process activities
                for activity in activities:
                    if not activity_types or activity['type'] in activity_types:
                        # Determine if it's a stroller run
                        activity_type = activity['type']
                        if activity_type == 'Run' and 'Ausflug' in activity['name']:
                            activity_type = 'RunStroller'
                            
                        all_activities.append({
                            'date': datetime.strptime(activity['start_date'][:10], '%Y-%m-%d'),
                            'distance': activity['distance'] / 1000,  # Convert to km
                            'name': activity['name'],
                            'type': activity_type,  # Use modified type
                            'moving_time': activity['moving_time'] / 3600,  # Convert to hours
                            'average_speed': (activity['distance'] / 1000) / (activity['moving_time'] / 3600),  # km/h,
                            'total_elevation_gain': activity['total_elevation_gain'] # m
                        })
                
                if len(activities) < 200:  # Last page
                    break
                    
                page += 1
                
            except Exception as e:
                print(f"Error getting activities: {e}")
                break
    
    # Cache the results before returning
    cache_activities(after, before, activity_types, all_activities)
    
    return all_activities

def calculate_performance_score(activity):
    """Calculate a performance score based on speed, distance, and duration."""
    # Base score is average speed * distance (rewards both speed and distance)
    base_score = activity['average_speed'] * activity['distance']
    
    # Add duration factor (longer activities get a small boost)
    duration_factor = 1 + (activity['moving_time'] / 4)  # 4 hours would double the score
    
    # Activity type specific multipliers to normalize different activities
    type_multipliers = {
        'Run': 1.0,
        'Ride': 0.05,  # Significantly reduced from 0.25 to account for easier gains
        'Swim': 4.0,   # Swimming is slower, so we increase the score
        'Walk': 2.0,   # Walking is slower
        'Hike': 1.5,   # Hiking is moderate
        'NordicSki': 0.8  # Nordic skiing is between running and biking
    }
    
    multiplier = type_multipliers.get(activity['type'], 1.0)
    
    return base_score * duration_factor * multiplier

def identify_activity_periods(total_data):
    """Identify distinct activity periods based on significant shifts in performance."""
    periods = []
    
    # Calculate the rate of change in the rolling average
    total_data['performance_change'] = total_data['rolling_performance'].diff(periods=28)
    
    # Normalize the change relative to the rolling average
    total_data['relative_change'] = total_data['performance_change'] / total_data['rolling_performance'].rolling(window=28, min_periods=1).mean()
    
    # Find significant shifts (where relative change exceeds threshold)
    threshold = 0.5  # Increased from 0.3 to 0.5 (50% change in rolling average)
    shift_points = total_data[abs(total_data['relative_change']) > threshold].index.tolist()
    
    if shift_points:
        current_start = 0
        for point in shift_points:
            # Only create a new period if it's been at least 30 days
            if point - current_start >= 30:
                # Find max performance and corresponding activity in this period
                period_data = total_data.iloc[current_start:point]
                max_avg = period_data['rolling_performance'].max()
                max_perf_idx = period_data['performance_score'].idxmax()
                max_perf = period_data.loc[max_perf_idx, 'performance_score']
                max_perf_date = period_data.loc[max_perf_idx, 'date']
                max_perf_name = period_data.loc[max_perf_idx, 'name']
                
                periods.append({
                    'start': total_data.iloc[current_start]['date'],
                    'end': total_data.iloc[point]['date'],
                    'max_avg': max_avg,
                    'max_performance': max_perf,
                    'max_performance_date': max_perf_date,
                    'max_performance_name': max_perf_name
                })
                current_start = point
        
        # Add the final period
        period_data = total_data.iloc[current_start:]
        max_avg = period_data['rolling_performance'].max()
        max_perf_idx = period_data['performance_score'].idxmax()
        max_perf = period_data.loc[max_perf_idx, 'performance_score']
        max_perf_date = period_data.loc[max_perf_idx, 'date']
        max_perf_name = period_data.loc[max_perf_idx, 'name']
        
        periods.append({
            'start': total_data.iloc[current_start]['date'],
            'end': total_data.iloc[-1]['date'],
            'max_avg': max_avg,
            'max_performance': max_perf,
            'max_performance_date': max_perf_date,
            'max_performance_name': max_perf_name
        })
    
    return periods

@app.route('/login')
def login():
    return redirect(f'https://www.strava.com/oauth/authorize?client_id={CLIENT_ID}'
                   f'&response_type=code&redirect_uri={request.host_url}callback'
                   f'&scope=activity:read_all')
@app.route('/calendar')
def get_calendar_data():
    after = request.args.get('after')
    before = request.args.get('before')
    
    token = get_token()
    if not token:
        return jsonify({'error': 'Could not get access token'})
        
    activities = get_activities(token, after, before, None)  # Get all activity types
    
    if activities:
        df = pd.DataFrame(activities)
        df['date'] = pd.to_datetime(df['date'])
        
        calendar_data = []
        for date, group in df.groupby('date'):
            day_activities = []
            for _, activity in group.iterrows():
                hours = int(activity['moving_time'])
                minutes = int((activity['moving_time'] - hours) * 60)
                day_activities.append({
                    'type': activity['type'],
                    'duration': f"{hours}h {minutes:02d}m",
                    'moving_time': activity['moving_time'],
                    'distance': activity['distance'] if (activity['type'] != 'WeightTraining' and activity['type'] != 'Workout') else 0,
                    'color': colors.get(activity['type'], '#999999'),
                    'total_elevation_gain': activity['total_elevation_gain'],
                    'name': activity['name']
                })
            
            calendar_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'activities': sorted(day_activities, key=lambda x: x['moving_time'], reverse=True)[:2]
            })
            
        return jsonify({
            'calendar': calendar_data,
            'colors': colors
        })
    
    return jsonify({'error': 'No activities found'})

@app.route('/callback')
def callback():
    if 'error' in request.args:
        return f"Error: {request.args.get('error')}"
    
    code = request.args.get('code')
    token_response = requests.post(
        'https://www.strava.com/oauth/token',
        data={
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET,
            'code': code,
            'grant_type': 'authorization_code'
        }
    )
    
    token_data = token_response.json()
    session['refresh_token'] = token_data.get('refresh_token')
    session['access_token'] = token_data.get('access_token')
    
    return redirect(url_for('index'))

@app.route('/')
def index():
    if 'refresh_token' not in session:
        return render_template('login.html')
    return render_template('index.html')

def get_token():
    """
    Retrieve a fresh access token using the stored refresh token.
    
    Returns:
        str: A new access token, or None if token retrieval fails.
    """
    # Check if refresh token exists in session or environment
    refresh_token = session.get('refresh_token') or os.getenv('STRAVA_REFRESH_TOKEN')
    
    if not refresh_token:
        print("No refresh token available")
        return None

    auth_url = "https://www.strava.com/oauth/token"
    payload = {
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token
    }
    
    try:
        response = requests.post(auth_url, data=payload)
        
        # Check if the request was successful
        if response.status_code != 200:
            print(f"Token refresh failed with status code {response.status_code}")
            print(f"Response content: {response.text}")
            return None
        
        token_data = response.json()
        
        # Validate the response
        if 'access_token' not in token_data:
            print("No access token in response")
            print(f"Response: {token_data}")
            return None
        
        # Update session with new tokens if using session-based auth
        if 'session' in locals() or 'session' in globals():
            session['access_token'] = token_data['access_token']
            session['refresh_token'] = token_data.get('refresh_token', refresh_token)
        
        return token_data['access_token']
    
    except requests.RequestException as e:
        print(f"Request error during token refresh: {e}")
        return None
    except ValueError as e:
        print(f"JSON parsing error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error during token refresh: {e}")
        return None
    
@app.route('/activities')
def get_activities_data():
    after = request.args.get('after')
    before = request.args.get('before')
    types = request.args.get('types', 'Run').split(',')
    
    token = get_token()
    if not token:
        return jsonify({'error': 'Could not get access token'})
        
    activities = get_activities(token, after, before, types)

    if activities:
        df = pd.DataFrame(activities)
        
        # Ensure date column is in datetime format
        df['date'] = pd.to_datetime(df['date'])

        # Group by date and type
        daily_distance = df.groupby(['date', 'type'])['distance'].sum().reset_index()
        
        # Ensure dates are sorted and aligned
        all_dates = sorted(daily_distance['date'].dt.strftime('%Y-%m-%d').unique().tolist())
        
        # Pivot table to align data properly
        pivot_df = daily_distance.pivot_table(index='date', columns='type', values='distance', fill_value=0)

        datasets = []

        for activity_type in pivot_df.columns:
            datasets.append({
                'label': f'{activity_type} Distance (km)',
                'data': pivot_df[activity_type].round(2).tolist(),
                'backgroundColor': colors.get(activity_type, '#999999'),
            })

        # Calculate total distance for the year
        yearly_total = df['distance'].sum()
        yearly_goal = float(os.getenv('YEARLY_GOAL', 1000))
        progress_percentage = (yearly_total / yearly_goal) * 100

        data = {
            'dates': all_dates,
            'datasets': datasets,
            'yearlyTotal': round(yearly_total, 2),
            'yearlyGoal': yearly_goal,
            'progressPercentage': round(progress_percentage, 1)
        }

        return jsonify(data)

    return jsonify({'error': 'No activities found'})

@app.route('/performance')
def get_performance_data():
    after = request.args.get('after')
    before = request.args.get('before')
    activity_types = request.args.get('types', 'Run').split(',')
    
    token = get_token()
    if not token:
        return jsonify({'error': 'Could not get access token'})
        
    activities = get_activities(token, after, before, activity_types)
    
    if activities:
        # Create DataFrame from all activities
        df = pd.DataFrame(activities)
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate performance score for each activity
        df['performance_score'] = df.apply(calculate_performance_score, axis=1)
        
        # Sort by date
        df = df.sort_values('date')
        
        # Create a date range from min to max date
        if after and before:
            date_range = pd.date_range(start=after, end=before, freq='D')
        else:
            date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
        
        # Calculate total performance first
        # For each date, find the sum of performance scores and the name of the activity with max score
        daily_scores = []
        for date, group in df.groupby('date'):
            max_perf_idx = group['performance_score'].idxmax()
            daily_scores.append({
                'date': date,
                'performance_score': group['performance_score'].sum(),
                'name': group.loc[max_perf_idx, 'name']
            })
        
        daily_total_scores = pd.DataFrame(daily_scores)
        
        daily_total_data = pd.DataFrame({'date': date_range})
        daily_total_data = daily_total_data.merge(daily_total_scores, on='date', how='left')
        daily_total_data['performance_score'] = daily_total_data['performance_score'].fillna(0)
        daily_total_data['name'] = daily_total_data['name'].fillna('')
        
        # Calculate rolling performance before identifying periods
        daily_total_data['rolling_performance'] = daily_total_data['performance_score'].rolling(
            window=28,
            min_periods=1
        ).mean()
        
        # Identify activity periods
        activity_periods = identify_activity_periods(daily_total_data)
        
        # Create background regions for periods
        background_regions = []
        for i, period in enumerate(activity_periods):
            background_regions.append({
                'type': 'box',
                'xMin': period['start'].strftime('%Y-%m-%d'),
                'xMax': period['end'].strftime('%Y-%m-%d'),
                'yMin': 0,
                'yMax': 'max',
                'backgroundColor': '#f5f5f5' if i % 2 == 0 else '#e0e0e0',
                'borderWidth': 0,
                'label': {
                    'content': f"Max 4-Week Avg: {period['max_avg']:.1f}\nHighest Performance: {period['max_performance']:.1f} ({period['max_performance_date'].strftime('%Y-%m-%d')})\nActivity: {period['max_performance_name']}",
                    'display': False,
                    'position': 'top'
                },
                'tooltipText': f"Max 4-Week Avg: {period['max_avg']:.1f}\nHighest Performance: {period['max_performance']:.1f} ({period['max_performance_date'].strftime('%Y-%m-%d')})\nActivity: {period['max_performance_name']}"
            })
        
        # Create datasets for the chart
        datasets = []
        
        # Add total performance first
        total_data = daily_total_data
        if not total_data.empty:
            # Daily total performance points
            non_zero_total = total_data[total_data['performance_score'] > 0]
            datasets.append({
                'label': 'Total Performance',
                'data': non_zero_total['performance_score'].round(2).tolist(),
                'dates': non_zero_total['date'].dt.strftime('%Y-%m-%d').tolist(),
                'type': 'scatter',
                'backgroundColor': '#000000',  # Black for total
                'borderColor': '#000000',
                'fill': False,
                'showLine': False
            })
            
            # Create smooth approximation curve using polynomial fit
            # Convert dates to numeric values for fitting
            x = np.arange(len(total_data))
            y = total_data['rolling_performance'].values
            
            # Fit a higher degree polynomial (8th degree for more flexibility)
            z = np.polyfit(x, y, 8)
            p = np.poly1d(z)
            
            # Generate smooth curve points with more points for smoother appearance
            smooth_x = np.linspace(0, len(total_data)-1, 200)
            smooth_y = p(smooth_x)
            
            # Convert back to dates for the smooth curve
            date_step = (len(total_data) - 1) / 199
            smooth_dates = [total_data['date'].iloc[int(i * date_step)].strftime('%Y-%m-%d') for i in range(200)]
            
            # Add the smooth approximation curve
            datasets.append({
                'label': 'Total Performance Trend',
                'data': smooth_y.round(2).tolist(),
                'dates': smooth_dates,
                'type': 'line',
                'backgroundColor': '#39FF14',  # Neon green
                'borderColor': '#39FF14',      # Neon green
                'fill': False,
                'borderWidth': 2,
                'tension': 0.2,
                'pointRadius': 0,
                'pointHoverRadius': 5
            })
            
            # Also add the 4-week rolling average
            datasets.append({
                'label': 'Total 4-Week Average',
                'data': total_data['rolling_performance'].round(2).tolist(),
                'dates': total_data['date'].dt.strftime('%Y-%m-%d').tolist(),
                'type': 'line',
                'backgroundColor': '#00000040',
                'borderColor': '#00000040',
                'fill': False,
                'borderDash': [5, 5]
            })
        
        # Add individual activity type performances
        for activity_type in activity_types:
            type_data = df[df['type'] == activity_type]
            if not type_data.empty:
                # Create daily data frame with all dates for this activity type
                daily_type_data = pd.DataFrame({'date': date_range})
                daily_type_scores = type_data.groupby('date')['performance_score'].sum().reset_index()
                daily_type_data = daily_type_data.merge(daily_type_scores, on='date', how='left')
                daily_type_data['performance_score'] = daily_type_data['performance_score'].fillna(0)
                
                # Calculate rolling average for this activity type
                daily_type_data['rolling_performance'] = daily_type_data['performance_score'].rolling(
                    window=28,
                    min_periods=1
                ).mean()
                
                # Daily performance points (only show non-zero values)
                non_zero_data = daily_type_data[daily_type_data['performance_score'] > 0]
                datasets.append({
                    'label': f'{activity_type} Performance',
                    'data': non_zero_data['performance_score'].round(2).tolist(),
                    'dates': non_zero_data['date'].dt.strftime('%Y-%m-%d').tolist(),
                    'type': 'scatter',
                    'backgroundColor': colors.get(activity_type, '#999999'),
                    'borderColor': colors.get(activity_type, '#999999'),
                    'fill': False,
                    'showLine': False
                })
                
                # 4-week rolling average
                datasets.append({
                    'label': f'{activity_type} 4-Week Average',
                    'data': daily_type_data['rolling_performance'].round(2).tolist(),
                    'dates': daily_type_data['date'].dt.strftime('%Y-%m-%d').tolist(),
                    'type': 'line',
                    'backgroundColor': colors.get(activity_type, '#999999') + '40',
                    'borderColor': colors.get(activity_type, '#999999') + '40',
                    'fill': False,
                    'borderDash': [5, 5]
                })
        
        return jsonify({
            'dates': date_range.strftime('%Y-%m-%d').tolist(),
            'datasets': datasets,
            'activityPeriods': background_regions
        })
    
    return jsonify({'error': 'No activities found'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)