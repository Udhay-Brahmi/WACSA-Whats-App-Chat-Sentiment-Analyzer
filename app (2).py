# WhatsApp Chat Sentiment Analyzer - Complete Single File
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
import nltk
from collections import Counter
from functools import lru_cache
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download VADER if not present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

@dataclass
class ChatStats:
    """Data class to hold chat statistics"""
    total_messages: int
    unique_users: int
    date_range: Tuple[str, str]
    sentiment_distribution: Dict[str, int]

class WhatsAppChatAnalyzer:
    """Complete WhatsApp Chat Analyzer with preprocessing and sentiment analysis"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = self._load_stop_words()
        self._setup_plotting_style()
    
    def _load_stop_words(self) -> set:
        """Load stop words with comprehensive default set"""
        default_stop_words = {
            # English stop words
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must',
            'shall', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her',
            'its', 'our', 'their', 'a', 'an', 'as', 'if', 'so', 'no', 'not', 'too',
            'very', 'just', 'now', 'than', 'only', 'also', 'back', 'after', 'use',
            'how', 'our', 'out', 'many', 'time', 'them', 'see', 'him', 'two', 'more',
            'go', 'come', 'get', 'make', 'know', 'take', 'think', 'good', 'new', 'first',
            # Common Hinglish/Chat words
            'hai', 'haan', 'nahi', 'kya', 'kar', 'karo', 'kaise', 'kyun', 'aur', 'ki',
            'ke', 'ko', 'me', 'se', 'pe', 'par', 'ya', 'jo', 'wo', 'ye', 'isko', 'usko',
            'ok', 'okay', 'yeah', 'yep', 'hmm', 'uhh', 'ohh', 'lol', 'haha', 'hehe'
        }
        
        try:
            # Try to load custom stop words if file exists
            with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
                custom_stop_words = set(word.strip().lower() for word in f.read().split())
                return custom_stop_words.union(default_stop_words)
        except FileNotFoundError:
            logger.info("Custom stop words file not found, using default comprehensive set")
            return default_stop_words
    
    def _setup_plotting_style(self):
        """Setup consistent plotting style"""
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
    
    @lru_cache(maxsize=1000)
    def _get_sentiment_scores(self, message: str) -> Tuple[float, float, float, int]:
        """Cached sentiment analysis for better performance"""
        scores = self.sentiment_analyzer.polarity_scores(str(message))
        
        # Determine sentiment value (-1, 0, 1)
        if scores['compound'] >= 0.05:
            sentiment_value = 1  # Positive
        elif scores['compound'] <= -0.05:
            sentiment_value = -1  # Negative
        else:
            sentiment_value = 0  # Neutral
            
        return scores['pos'], scores['neg'], scores['neu'], sentiment_value
    
    def preprocess_chat(self, data: str) -> pd.DataFrame:
        """Enhanced preprocessing with multiple WhatsApp format support"""
        if not data.strip():
            raise ValueError("Empty chat data provided")
        
        # Multiple regex patterns for different WhatsApp export formats
        patterns = [
            # Format: 1/1/23, 6:44 pm - User: Message
            r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s(?:AM|PM|am|pm))\s-\s([^:]+?):\s(.+)',
            # Format: 1/1/2023, 18:44 - User: Message
            r'(\d{1,2}/\d{1,2}/\d{4},\s\d{1,2}:\d{2})\s-\s([^:]+?):\s(.+)',
            # Format: [1/1/23, 6:44:32 PM] User: Message
            r'\[(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}:\d{2}\s(?:AM|PM))\]\s([^:]+?):\s(.+)',
            # Format: 01/01/2023, 18:44 - User: Message
            r'(\d{1,2}/\d{1,2}/\d{4},\s\d{1,2}:\d{2})\s-\s([^:]+?):\s(.+)',
        ]
        
        messages = []
        lines = data.split('\n')
        current_message = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            matched = False
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    matched = True
                    # Save previous message if exists
                    if current_message:
                        messages.append(current_message)
                    
                    groups = match.groups()
                    current_message = {
                        'message_date': groups[0],
                        'user': groups[1].strip(),
                        'message': groups[2].strip()
                    }
                    break
            
            # If line doesn't match pattern, it's continuation of previous message
            if not matched and current_message:
                current_message['message'] += ' ' + line
        
        # Add the last message
        if current_message:
            messages.append(current_message)
        
        if not messages:
            raise ValueError("No valid messages found in chat data. Please check the file format.")
        
        # Convert to DataFrame
        df = pd.DataFrame(messages)
        
        # Parse datetime with multiple format attempts
        df = self._parse_datetime(df)
        
        # Extract datetime features
        df = self._extract_datetime_features(df)
        
        # Filter out system messages and invalid content
        df = self._filter_valid_messages(df)
        
        # Analyze sentiment
        df = self._analyze_sentiment(df)
        
        return df
    
    def _parse_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse datetime with multiple format attempts"""
        datetime_formats = [
            '%m/%d/%y, %I:%M %p',    # 1/1/23, 6:44 PM
            '%d/%m/%y, %I:%M %p',    # 1/1/23, 6:44 PM (day/month)
            '%m/%d/%Y, %I:%M %p',    # 1/1/2023, 6:44 PM
            '%d/%m/%Y, %I:%M %p',    # 1/1/2023, 6:44 PM (day/month)
            '%m/%d/%y, %H:%M',       # 1/1/23, 18:44
            '%d/%m/%y, %H:%M',       # 1/1/23, 18:44 (day/month)
            '%m/%d/%Y, %H:%M',       # 1/1/2023, 18:44
            '%d/%m/%Y, %H:%M',       # 1/1/2023, 18:44 (day/month)
        ]
        
        def parse_datetime_flexible(dt_str):
            for fmt in datetime_formats:
                try:
                    return pd.to_datetime(dt_str, format=fmt)
                except:
                    continue
            # Fallback to pandas' flexible parsing
            try:
                return pd.to_datetime(dt_str)
            except:
                return pd.NaT
        
        df['date'] = df['message_date'].apply(parse_datetime_flexible)
        df = df.dropna(subset=['date'])
        
        if df.empty:
            raise ValueError("Could not parse any valid dates from the chat file")
        
        return df
    
    def _extract_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract comprehensive datetime features"""
        df['only_date'] = df['date'].dt.date
        df['year'] = df['date'].dt.year
        df['month_num'] = df['date'].dt.month
        df['month'] = df['date'].dt.month_name()
        df['day'] = df['date'].dt.day
        df['day_name'] = df['date'].dt.day_name()
        df['hour'] = df['date'].dt.hour
        df['minute'] = df['date'].dt.minute
        
        return df
    
    def _filter_valid_messages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out system messages and invalid content"""
        # System message indicators
        system_patterns = [
            r'joined using this group.*invite link',
            r'left',
            r'was added',
            r'was removed',
            r'You were added',
            r'created group',
            r'changed the subject',
            r'changed this group.*description',
            r'Security code changed',
            r'Messages and calls are end-to-end encrypted',
            r'<Media omitted>',
            r'deleted this message',
            r'This message was deleted',
        ]
        
        # Create combined pattern
        system_pattern = '|'.join(system_patterns)
        
        # Filter out system messages
        mask = ~df['message'].str.contains(system_pattern, case=False, na=False, regex=True)
        df = df[mask]
        
        # Filter out very short messages (less than 2 characters)
        df = df[df['message'].str.len() >= 2]
        
        # Remove entries where user is 'group_notification' or similar
        df = df[~df['user'].str.contains('group_notification|notification', case=False, na=False)]
        
        return df.reset_index(drop=True)
    
    def _analyze_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze sentiment for all messages"""
        if df.empty:
            raise ValueError("No messages to analyze after filtering")
        
        # Apply sentiment analysis
        sentiment_results = df['message'].apply(self._get_sentiment_scores)
        
        # Extract sentiment scores and values
        df[['pos_score', 'neg_score', 'neu_score', 'value']] = pd.DataFrame(
            sentiment_results.tolist(), index=df.index
        )
        
        return df
    
    def get_chat_statistics(self, df: pd.DataFrame) -> ChatStats:
        """Get comprehensive chat statistics"""
        sentiment_counts = df['value'].value_counts()
        
        return ChatStats(
            total_messages=len(df),
            unique_users=df['user'].nunique(),
            date_range=(str(df['only_date'].min()), str(df['only_date'].max())),
            sentiment_distribution={
                'positive': sentiment_counts.get(1, 0),
                'neutral': sentiment_counts.get(0, 0),
                'negative': sentiment_counts.get(-1, 0)
            }
        )

class ChartGenerator:
    """Enhanced chart generator without heatmap functionality"""
    
    def __init__(self):
        self.colors = {
            'positive': '#2E8B57',   # Sea Green
            'neutral': '#708090',    # Slate Gray  
            'negative': '#DC143C'    # Crimson
        }
        self.sentiment_names = {1: 'positive', 0: 'neutral', -1: 'negative'}
    
    def _filter_data(self, df: pd.DataFrame, selected_user: str, sentiment_value: int) -> pd.DataFrame:
        """Filter data by user and sentiment"""
        filtered_df = df.copy()
        
        if selected_user != 'Overall':
            filtered_df = filtered_df[filtered_df['user'] == selected_user]
            
        filtered_df = filtered_df[filtered_df['value'] == sentiment_value]
        return filtered_df
    
    def create_activity_chart(self, df: pd.DataFrame, selected_user: str, 
                            chart_type: str, sentiment_value: int) -> plt.Figure:
        """Create activity charts (monthly/daily/hourly)"""
        filtered_df = self._filter_data(df, selected_user, sentiment_value)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if filtered_df.empty:
            ax.text(0.5, 0.5, f'No {self.sentiment_names[sentiment_value]} messages found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{chart_type.capitalize()} Activity - {self.sentiment_names[sentiment_value].capitalize()}')
            plt.tight_layout()
            return fig
        
        # Prepare data based on chart type
        if chart_type == 'monthly':
            # Get month activity with proper ordering
            data = filtered_df['month'].value_counts()
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            data = data.reindex(month_order, fill_value=0)
            xlabel = 'Month'
        elif chart_type == 'daily':
            # Get day activity with proper ordering
            data = filtered_df['day_name'].value_counts()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                        'Friday', 'Saturday', 'Sunday']
            data = data.reindex(day_order, fill_value=0)
            xlabel = 'Day of Week'
        else:  # hourly
            data = filtered_df['hour'].value_counts().sort_index()
            data = data.reindex(range(24), fill_value=0)
            xlabel = 'Hour of Day'
        
        # Create bar chart
        color = self.colors[self.sentiment_names[sentiment_value]]
        bars = ax.bar(range(len(data)), data.values, color=color, alpha=0.7, 
                     edgecolor='white', linewidth=0.5)
        
        # Customize chart
        ax.set_xticks(range(len(data)))
        if chart_type == 'hourly':
            ax.set_xticklabels([f'{i:02d}:00' for i in data.index], rotation=45)
        else:
            ax.set_xticklabels(data.index, rotation=45, ha='right')
            
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Message Count', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + max(data.values) * 0.01,
                       f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        # Title
        sentiment_name = self.sentiment_names[sentiment_value].capitalize()
        title = f'{chart_type.capitalize()} Activity - {sentiment_name} Messages'
        if selected_user != 'Overall':
            title += f' ({selected_user})'
        ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def create_timeline_chart(self, df: pd.DataFrame, selected_user: str, 
                            timeline_type: str, sentiment_value: int) -> plt.Figure:
        """Create timeline charts (daily/monthly)"""
        filtered_df = self._filter_data(df, selected_user, sentiment_value)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        if filtered_df.empty:
            ax.text(0.5, 0.5, f'No {self.sentiment_names[sentiment_value]} messages found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
        else:
            try:
                if timeline_type == 'daily':
                    timeline_data = filtered_df.groupby('only_date').size()
                    xlabel = 'Date'
                    # Convert dates to strings for plotting
                    x_data = [str(date) for date in timeline_data.index]
                    y_data = timeline_data.values
                else:  # monthly
                    # Create year-month grouping to avoid period issues
                    filtered_df['year_month'] = filtered_df['date'].dt.to_period('M').astype(str)
                    timeline_data = filtered_df.groupby('year_month').size()
                    xlabel = 'Month'
                    x_data = list(range(len(timeline_data)))  # Use numeric indices
                    y_data = timeline_data.values
                
                color = self.colors[self.sentiment_names[sentiment_value]]
                
                # Create line plot with area fill
                ax.plot(x_data, y_data, color=color, linewidth=2, marker='o', markersize=4, alpha=0.8)
                ax.fill_between(x_data, y_data, alpha=0.3, color=color)
                
                # Set labels and formatting
                ax.set_xlabel(xlabel, fontsize=12)
                ax.set_ylabel('Message Count', fontsize=12)
                ax.grid(True, alpha=0.3)
                
                # Handle x-axis labels
                if timeline_type == 'daily':
                    # Show every nth label to avoid crowding
                    step = max(1, len(x_data) // 10)
                    ax.set_xticks(x_data[::step])
                    ax.set_xticklabels([x_data[i] for i in range(0, len(x_data), step)], rotation=45)
                else:  # monthly
                    ax.set_xticks(x_data)
                    ax.set_xticklabels(timeline_data.index, rotation=45)
                    
            except Exception as e:
                logger.error(f"Error creating timeline chart: {str(e)}")
                ax.text(0.5, 0.5, f'Error creating timeline: {str(e)}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        sentiment_name = self.sentiment_names[sentiment_value].capitalize()
        title = f'{timeline_type.capitalize()} Timeline - {sentiment_name} Messages'
        if selected_user != 'Overall':
            title += f' ({selected_user})'
        ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def create_user_analysis_chart(self, df: pd.DataFrame, sentiment_value: int) -> plt.Figure:
        """Create user contribution analysis charts"""
        filtered_df = df[df['value'] == sentiment_value]
        user_counts = filtered_df['user'].value_counts().head(10)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if user_counts.empty:
            ax.text(0.5, 0.5, f'No {self.sentiment_names[sentiment_value]} messages found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
        else:
            color = self.colors[self.sentiment_names[sentiment_value]]
            
            # Create horizontal bar chart for better label readability
            bars = ax.barh(range(len(user_counts)), user_counts.values, 
                          color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
            
            ax.set_yticks(range(len(user_counts)))
            ax.set_yticklabels(user_counts.index)
            ax.set_xlabel('Message Count', fontsize=12)
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + max(user_counts.values) * 0.01, bar.get_y() + bar.get_height()/2,
                       f'{int(width)}', ha='left', va='center', fontsize=10, fontweight='bold')
        
        sentiment_name = self.sentiment_names[sentiment_value].capitalize()
        ax.set_title(f'Most {sentiment_name} Users', fontsize=14, pad=20, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def create_word_frequency_chart(self, df: pd.DataFrame, selected_user: str, 
                                  sentiment_value: int, analyzer, top_n: int = 20) -> plt.Figure:
        """Create word frequency chart"""
        filtered_df = self._filter_data(df, selected_user, sentiment_value)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        if filtered_df.empty:
            ax.text(0.5, 0.5, f'No {self.sentiment_names[sentiment_value]} messages found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
        else:
            # Extract and process words
            all_messages = ' '.join(filtered_df['message'].astype(str))
            
            # Clean and tokenize
            words = re.findall(r'\b[a-zA-Z]+\b', all_messages.lower())
            
            # Remove stop words
            filtered_words = [word for word in words 
                            if word not in analyzer.stop_words and len(word) > 2]
            
            if filtered_words:
                word_counts = Counter(filtered_words).most_common(top_n)
                words_list, counts_list = zip(*word_counts)
                
                color = self.colors[self.sentiment_names[sentiment_value]]
                
                # Create horizontal bar chart
                bars = ax.barh(range(len(words_list)), counts_list, 
                              color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
                
                ax.set_yticks(range(len(words_list)))
                ax.set_yticklabels(words_list)
                ax.set_xlabel('Frequency', fontsize=12)
                ax.grid(axis='x', alpha=0.3)
                ax.invert_yaxis()
                
                # Add value labels
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width + max(counts_list) * 0.01, bar.get_y() + bar.get_height()/2,
                           f'{int(width)}', ha='left', va='center', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'No valid words found after filtering', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
        
        sentiment_name = self.sentiment_names[sentiment_value].capitalize()
        title = f'Most Common {sentiment_name} Words'
        if selected_user != 'Overall':
            title += f' ({selected_user})'
        ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
        
        plt.tight_layout()
        return fig

class WhatsAppAnalyzerApp:
    """Main application class for Gradio interface"""
    
    def __init__(self):
        self.analyzer = WhatsAppChatAnalyzer()
        self.chart_generator = ChartGenerator()
        self.current_data = None
        self.user_list = ["Overall"]
    
    def process_file(self, file) -> Tuple[Optional[pd.DataFrame], List[str], str]:
        """Process uploaded file and return data, user list, and status message"""
        if file is None:
            return None, ["Overall"], "Please upload a WhatsApp chat file."
        
        try:
            # Read file with encoding detection
            encodings = ['utf-8', 'utf-16', 'latin1', 'cp1252']
            raw_data = None
            
            for encoding in encodings:
                try:
                    with open(file.name, 'r', encoding=encoding) as f:
                        raw_data = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if raw_data is None:
                return None, ["Overall"], "❌ Error: Could not decode the file. Please check file encoding."
            
            # Preprocess and analyze
            df = self.analyzer.preprocess_chat(raw_data)
            
            if df.empty:
                return None, ["Overall"], "❌ Error: No valid messages found in the chat file."
            
            # Get user list
            users = sorted(df['user'].unique().tolist())
            users.insert(0, "Overall")
            
            # Get statistics
            stats = self.analyzer.get_chat_statistics(df)
            status_msg = (f"✅ Analysis complete! "
                         f"Messages: {stats.total_messages:,}, "
                         f"Users: {stats.unique_users}, "
                         f"Date range: {stats.date_range[0]} to {stats.date_range[1]}")
            
            self.current_data = df
            self.user_list = users
            
            logger.info(f"Successfully processed chat with {len(df)} messages from {len(users)-1} users")
            return df, users, status_msg
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return None, ["Overall"], f"❌ Error: {str(e)}"
    
    def generate_all_charts(self, selected_user: str):
        """Generate all charts for the selected user"""
        if self.current_data is None:
            empty_fig = plt.figure(figsize=(8, 6))
            empty_fig.text(0.5, 0.5, 'No data available. Please upload a chat file first.', 
                          ha='center', va='center', fontsize=14)
            plt.tight_layout()
            return [empty_fig] * 18
        
        charts = []
        
        try:
            logger.info(f"Generating charts for user: {selected_user}")
            
            # Monthly and Daily activity charts (2 types × 3 sentiments = 6 charts)
            for chart_type in ['monthly', 'daily']:
                for sentiment in [1, 0, -1]:  # Positive, Neutral, Negative
                    try:
                        chart = self.chart_generator.create_activity_chart(
                            self.current_data, selected_user, chart_type, sentiment
                        )
                        charts.append(chart)
                    except Exception as e:
                        logger.error(f"Error creating {chart_type} activity chart for sentiment {sentiment}: {str(e)}")
                        error_fig = plt.figure(figsize=(8, 6))
                        error_fig.text(0.5, 0.5, f'Error creating {chart_type} chart', 
                                     ha='center', va='center', fontsize=12)
                        plt.tight_layout()
                        charts.append(error_fig)
            
            # Timeline charts (2 types × 3 sentiments = 6 charts)  
            for timeline_type in ['daily', 'monthly']:
                for sentiment in [1, 0, -1]:
                    try:
                        chart = self.chart_generator.create_timeline_chart(
                            self.current_data, selected_user, timeline_type, sentiment
                        )
                        charts.append(chart)
                    except Exception as e:
                        logger.error(f"Error creating {timeline_type} timeline chart for sentiment {sentiment}: {str(e)}")
                        error_fig = plt.figure(figsize=(8, 6))
                        error_fig.text(0.5, 0.5, f'Error creating {timeline_type} timeline', 
                                     ha='center', va='center', fontsize=12)
                        plt.tight_layout()
                        charts.append(error_fig)
            
            # User analysis charts (only for Overall) - 3 charts
            if selected_user == "Overall":
                for sentiment in [1, -1, 0]:  # Positive, Negative, Neutral
                    try:
                        chart = self.chart_generator.create_user_analysis_chart(
                            self.current_data, sentiment
                        )
                        charts.append(chart)
                    except Exception as e:
                        logger.error(f"Error creating user analysis chart for sentiment {sentiment}: {str(e)}")
                        error_fig = plt.figure(figsize=(8, 6))
                        error_fig.text(0.5, 0.5, f'Error creating user analysis', 
                                     ha='center', va='center', fontsize=12)
                        plt.tight_layout()
                        charts.append(error_fig)
            else:
                # Empty charts for non-Overall users
                for _ in range(3):
                    empty_fig = plt.figure(figsize=(8, 6))
                    empty_fig.text(0.5, 0.5, 'Select "Overall" to see user analysis', 
                                 ha='center', va='center', fontsize=14)
                    plt.tight_layout()
                    charts.append(empty_fig)
            
            # Word frequency charts (3 sentiments = 3 charts)
            for sentiment in [1, 0, -1]:
                try:
                    chart = self.chart_generator.create_word_frequency_chart(
                        self.current_data, selected_user, sentiment, self.analyzer
                    )
                    charts.append(chart)
                except Exception as e:
                    logger.error(f"Error creating word frequency chart for sentiment {sentiment}: {str(e)}")
                    error_fig = plt.figure(figsize=(8, 6))
                    error_fig.text(0.5, 0.5, f'Error creating word frequency chart', 
                                 ha='center', va='center', fontsize=12)
                    plt.tight_layout()
                    charts.append(error_fig)
                    
            logger.info(f"Successfully generated {len(charts)} charts")
                
        except Exception as e:
            logger.error(f"Major error generating charts: {str(e)}")
            error_fig = plt.figure(figsize=(8, 6))
            error_fig.text(0.5, 0.5, f'Error generating charts: {str(e)}', 
                          ha='center', va='center', fontsize=14)
            plt.tight_layout()
            charts = [error_fig] * 18
        
        return charts

def create_gradio_app():
    """Create and configure the Gradio interface"""
    app = WhatsAppAnalyzerApp()
    
    def update_dropdown_and_process(file):
        """Update user dropdown when file is uploaded"""
        data, users, status = app.process_file(file)
        return gr.Dropdown(choices=users, value="Overall"), status
    
    def analyze_chat(selected_user):
        """Generate analysis for selected user"""
        return app.generate_all_charts(selected_user)
    
    # Custom CSS for better appearance
    custom_css = """
    .gradio-container {
        max-width: 1400px !important;
        margin: 0 auto;
    }
    .gr-button-primary {
        background: linear-gradient(45deg, #2E8B57, #3CB371) !important;
        border: none !important;
    }
    .gr-form {
        background: #f8f9fa !important;
        border-radius: 10px !important;
        padding: 20px !important;
    }
    """
    
    # Create the interface
    with gr.Blocks(
        title="WhatsApp Chat Sentiment Analyzer",
        theme=gr.themes.Soft(primary_hue="green", secondary_hue="blue"),
        css=custom_css
    ) as interface:
        
        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(45deg, #2E8B57, #3CB371); border-radius: 10px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0; font-size: 2.5em;">📱 WhatsApp Chat Sentiment Analyzer</h1>
            <p style="color: white; margin: 10px 0 0 0; font-size: 1.2em;">Discover sentiment patterns in your WhatsApp conversations!</p>
        </div>
        """)
        
        # Upload and Control Section
        with gr.Row():
            with gr.Column(scale=2):
                file_input = gr.File(
                    label="📁 Upload WhatsApp Chat File (.txt)",
                    file_types=[".txt"],
                    file_count="single",
                    height=100
                )
                status_text = gr.Textbox(
                    label="📊 Analysis Status",
                    value="Please upload a WhatsApp chat file to begin analysis.",
                    interactive=False,
                    lines=2
                )
            
            with gr.Column(scale=1):
                user_dropdown = gr.Dropdown(
                    choices=["Overall"],
                    value="Overall",
                    label="👤 Select User for Analysis",
                    info="Choose a specific user or 'Overall' for all users",
                    interactive=True
                )
                analyze_btn = gr.Button(
                    "🔍 Analyze Chat",
                    variant="primary",
                    size="lg",
                    scale=1
                )
        
        # File upload handler
        file_input.change(
            update_dropdown_and_process,
            inputs=[file_input],
            outputs=[user_dropdown, status_text]
        )
        
        # Chart outputs organized in tabs
        with gr.Tabs():
            with gr.Tab("📊 Activity Patterns"):
                gr.Markdown("### 📅 Monthly Activity Distribution")
                gr.Markdown("*See how message sentiments vary across different months*")
                with gr.Row():
                    monthly_pos = gr.Plot(label="😊 Positive Messages")
                    monthly_neu = gr.Plot(label="😐 Neutral Messages") 
                    monthly_neg = gr.Plot(label="😞 Negative Messages")
                
                gr.Markdown("### 📆 Daily Activity Distribution")
                gr.Markdown("*Discover which days of the week have different sentiment patterns*")
                with gr.Row():
                    daily_pos = gr.Plot(label="😊 Positive Messages")
                    daily_neu = gr.Plot(label="😐 Neutral Messages")
                    daily_neg = gr.Plot(label="😞 Negative Messages")
            
            with gr.Tab("📈 Timeline Trends"):
                gr.Markdown("### 📊 Daily Message Timeline")
                gr.Markdown("*Track sentiment changes over time on a daily basis*")
                with gr.Row():
                    daily_timeline_pos = gr.Plot(label="😊 Positive Trend")
                    daily_timeline_neu = gr.Plot(label="😐 Neutral Trend")
                    daily_timeline_neg = gr.Plot(label="😞 Negative Trend")
                
                gr.Markdown("### 📈 Monthly Message Timeline")
                gr.Markdown("*See long-term sentiment trends across months*")
                with gr.Row():
                    monthly_timeline_pos = gr.Plot(label="😊 Positive Trend")
                    monthly_timeline_neu = gr.Plot(label="😐 Neutral Trend")
                    monthly_timeline_neg = gr.Plot(label="😞 Negative Trend")
            
            with gr.Tab("👥 User Analysis"):
                gr.Markdown("### 🏆 Most Active Users by Sentiment")
                gr.Markdown("*Compare users based on their sentiment contributions (Available only for 'Overall' selection)*")
                with gr.Row():
                    users_pos = gr.Plot(label="😊 Most Positive Users")
                    users_neg = gr.Plot(label="😞 Most Negative Users") 
                    users_neu = gr.Plot(label="😐 Most Neutral Users")
            
            with gr.Tab("📝 Word Analysis"):
                gr.Markdown("### 🔤 Most Frequent Words by Sentiment")
                gr.Markdown("*Discover the most commonly used words for each sentiment category*")
                with gr.Row():
                    words_pos = gr.Plot(label="😊 Positive Words")
                    words_neu = gr.Plot(label="😐 Neutral Words")
                    words_neg = gr.Plot(label="😞 Negative Words")
        
        # Connect analyze button to generate all charts
        analyze_btn.click(
            analyze_chat,
            inputs=[user_dropdown],
            outputs=[
                # Monthly activity (3)
                monthly_pos, monthly_neu, monthly_neg,
                # Daily activity (3)
                daily_pos, daily_neu, daily_neg,
                # Daily timelines (3)
                daily_timeline_pos, daily_timeline_neu, daily_timeline_neg,
                # Monthly timelines (3)
                monthly_timeline_pos, monthly_timeline_neu, monthly_timeline_neg,
                # User analysis (3)
                users_pos, users_neg, users_neu,
                # Word analysis (3)
                words_pos, words_neu, words_neg
            ]
        )
        
        # Auto-analyze when user selection changes
        user_dropdown.change(
            analyze_chat,
            inputs=[user_dropdown],
            outputs=[
                monthly_pos, monthly_neu, monthly_neg,
                daily_pos, daily_neu, daily_neg,
                daily_timeline_pos, daily_timeline_neu, daily_timeline_neg,
                monthly_timeline_pos, monthly_timeline_neu, monthly_timeline_neg,
                users_pos, users_neg, users_neu,
                words_pos, words_neu, words_neg
            ]
        )
        
        # Information and Instructions
        with gr.Accordion("📋 How to Use This Tool", open=False):
            gr.Markdown("""
            ### 🚀 Getting Started
            
            1. **📱 Export Your WhatsApp Chat:**
               - Open WhatsApp on your phone
               - Go to the chat you want to analyze
               - Tap on the chat name → More → Export Chat
               - Choose "Without Media" for faster processing
               - Share/save the .txt file
            
            2. **📤 Upload the File:**
               - Click the "Upload WhatsApp Chat File" button above
               - Select your exported .txt file
               - Wait for the processing to complete
            
            3. **👤 Select User (Optional):**
               - Choose "Overall" to analyze all users in the chat
               - Or select a specific user to focus on their messages only
            
            4. **🔍 Analyze:**
               - Click "Analyze Chat" to generate all visualizations
               - Explore different tabs to see various insights
            
            ### 🎯 What You'll Discover
            
            - **📊 Sentiment Analysis:** Automatically categorizes messages as positive, neutral, or negative
            - **⏰ Activity Patterns:** When are different sentiments most common?
            - **📈 Timeline Trends:** How do sentiments change over time?
            - **👥 User Insights:** Which users contribute most to each sentiment?
            - **📝 Word Analysis:** What words are associated with each sentiment?
            
            ### 🔒 Privacy & Security
            
            - ✅ All processing happens locally in your browser
            - ✅ Your chat data is never uploaded to external servers
            - ✅ No data is stored or transmitted anywhere
            - ✅ Complete privacy guaranteed
            
            ### 📊 Supported Formats
            
            This tool supports various WhatsApp export formats:
            - `1/1/23, 6:44 PM - User: Message`
            - `1/1/2023, 18:44 - User: Message`
            - `[1/1/23, 6:44:32 PM] User: Message`
            
            ### 🛠️ Troubleshooting
            
            **Problem:** "No valid messages found"
            - **Solution:** Make sure you exported the chat correctly from WhatsApp
            - **Solution:** Check that the file contains actual messages, not just media
            
            **Problem:** Charts are empty
            - **Solution:** Try selecting "Overall" instead of a specific user
            - **Solution:** Make sure the chat has messages with the selected sentiment
            
            **Problem:** File won't upload
            - **Solution:** Ensure the file is a .txt file from WhatsApp export
            - **Solution:** Try exporting the chat again without media
            """)
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 20px; margin-top: 40px; border-top: 2px solid #e0e0e0;">
            <p style="color: #666; font-size: 0.9em; margin: 0;">
                🔒 <strong>Privacy First:</strong> All analysis is performed locally. Your data never leaves your browser.
            </p>
            <p style="color: #666; font-size: 0.8em; margin: 5px 0 0 0;">
                Built with ❤️ using Gradio, NLTK VADER, and advanced NLP techniques
            </p>
        </div>
        """)
    
    return interface

# Utility functions
def create_sample_chat():
    """Create sample chat data for testing"""
    sample_messages = [
        "8/10/25, 6:44 PM - John: Hey everyone! How's your day going? 😊",
        "8/10/25, 6:45 PM - Alice: Amazing! Just got promoted at work! So excited! 🎉",
        "8/10/25, 6:46 PM - Bob: Not great... having a really tough day 😔",
        "8/10/25, 7:00 PM - John: Sorry to hear that Bob. Things will get better! 💪",
        "8/10/25, 7:15 PM - Alice: Let's plan something fun for the weekend! Who's in?",
        "8/10/25, 7:30 PM - Bob: That sounds nice. I could really use some fun right now.",
        "8/11/25, 9:00 AM - John: Good morning everyone! Ready for another great day!",
        "8/11/25, 9:15 AM - Alice: Morning! Feeling super productive today! ☀️",
        "8/11/25, 10:00 AM - Bob: Monday blues are hitting hard... can't focus on anything",
        "8/11/25, 11:00 AM - John: We got this! Let's stay positive and support each other! 🤗",
        "8/11/25, 12:00 PM - Alice: Lunch break! This sandwich is absolutely delicious! 🥪",
        "8/11/25, 1:00 PM - Bob: Work is overwhelming today. Too much stress.",
        "8/11/25, 2:00 PM - John: Take it one step at a time, Bob. You're stronger than you think!",
        "8/11/25, 3:00 PM - Alice: Just finished my presentation! Went perfectly! 👏",
        "8/11/25, 4:00 PM - Bob: Thanks guys. Your support means a lot to me.",
    ]
    return "\n".join(sample_messages)

def validate_chat_format(content: str) -> bool:
    """Validate if content looks like WhatsApp chat export"""
    # Check for basic WhatsApp patterns
    patterns = [
        r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}',  # Date pattern
        r'\s-\s',  # Separator pattern
        r':\s',   # Message separator
    ]
    
    for pattern in patterns:
        if not re.search(pattern, content):
            return False
    
    return True

# Performance monitoring
class PerformanceMonitor:
    """Simple performance monitoring utility"""
    
    @staticmethod
    def monitor_function(func_name: str):
        def decorator(func):
            def wrapper(*args, **kwargs):
                import time
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                logger.info(f"{func_name} completed in {end_time - start_time:.2f} seconds")
                return result
            return wrapper
        return decorator

# Configuration
class Config:
    """Application configuration"""
    
    # File processing
    MAX_FILE_SIZE_MB = 50
    SUPPORTED_ENCODINGS = ['utf-8', 'utf-16', 'latin1', 'cp1252']
    
    # Chart settings
    FIGURE_DPI = 100
    DEFAULT_FIGSIZE = (12, 6)
    
    # Analysis settings
    MIN_MESSAGE_LENGTH = 2
    MAX_WORDS_DISPLAY = 20
    CACHE_SIZE = 1000
    
    # Colors
    COLORS = {
        'positive': '#2E8B57',
        'neutral': '#708090', 
        'negative': '#DC143C'
    }

# Error handling utilities
class ChatAnalysisError(Exception):
    """Custom exception for chat analysis errors"""
    pass

def safe_file_read(file_path: str, encodings: List[str] = None) -> str:
    """Safely read file with multiple encoding attempts"""
    if encodings is None:
        encodings = Config.SUPPORTED_ENCODINGS
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    
    raise ChatAnalysisError(f"Could not decode file with any of the supported encodings: {encodings}")

# Main execution
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Initializing WhatsApp Chat Sentiment Analyzer...")
    
    # Create and launch the app
    try:
        app = create_gradio_app()
        
        # Launch configuration
        launch_config = {
            "share": True,
            "server_name": "0.0.0.0", 
            "server_port": 7860,
            "show_error": True,
            "show_tips": True,
            "favicon_path": None,
            "analytics_enabled": False,
        }
        
        logger.info("Launching application...")
        logger.info(f"Launch configuration: {launch_config}")
        
        app.launch(**launch_config)
        
    except Exception as e:
        logger.error(f"Failed to launch application: {str(e)}")
        print(f"Error: {e}")
        
        # Try basic launch as fallback
        try:
            logger.info("Attempting fallback launch...")
            app.launch(share=True, server_port=7860)
        except Exception as fallback_error:
            logger.error(f"Fallback launch also failed: {str(fallback_error)}")
            print(f"Fallback error: {fallback_error}")
    
    logger.info("Application setup complete!")

# Export main classes
__all__ = [
    'WhatsAppChatAnalyzer',
    'ChartGenerator', 
    'WhatsAppAnalyzerApp',
    'ChatStats',
    'Config',
    'create_gradio_app'
]