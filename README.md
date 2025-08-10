# WhatsApp Chat Sentiment Analyzer

Analyze your WhatsApp chat data to uncover sentiment trends, activity patterns, and user insights. This Gradio-based web application provides an intuitive interface to upload your chat export file and visualize various aspects of your conversations.

<img width="1556" height="910" alt="Image" src="https://github.com/user-attachments/assets/6cf5ee88-0308-4555-9516-53ea3df37bbb" />

---

## 💡 Project Overview

The WhatsApp Chat Sentiment Analyzer is a powerful, privacy-focused tool designed to help you understand the emotional tone and communication patterns within your WhatsApp chats. By simply uploading an exported `.txt` file of your chat history (without media), the application processes the messages and generates a series of interactive visualizations. It leverages **Natural Language Processing (NLP)** techniques, specifically **sentiment analysis**, to categorize messages as positive, neutral, or negative. Beyond sentiment, it also provides insights into chat activity over time, highlights key contributors, and identifies common words associated with different emotional tones. This tool is ideal for anyone looking to gain deeper insights into their personal or group conversations, all while ensuring **complete data privacy** as all analysis occurs directly in your browser.

---

## ✨ Features

* **📊 Sentiment Analysis:** This core feature uses **NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner)** lexicon to automatically classify each message's emotional tone. Messages are assigned a score that determines whether they are predominantly **positive 😊**, **neutral 😐**, or **negative 😞**. VADER is particularly effective for social media texts due to its sensitivity to both polarity and intensity.

* **⏰ Activity Patterns:** Gain insights into *when* different sentiments are expressed. This feature generates charts that show the distribution of positive, neutral, and negative messages across:

    * **Monthly Activity:** Reveals which months see the most activity for each sentiment.

    * **Daily Activity:** Highlights the days of the week where specific sentiments are more prevalent.

    * **Hourly Activity:** Pinpoints the hours of the day when certain emotional tones dominate conversations.

* **📈 Timeline Trends:** Observe how sentiment evolves over time. These charts illustrate the fluctuations in positive, neutral, and negative messages on both:

    * **Daily Timelines:** Providing a granular view of sentiment shifts day-by-day.

    * **Monthly Timelines:** Offering a broader perspective on long-term emotional trends in the chat.

* **👥 User Insights:** Understand the emotional contributions of individual participants. This analysis identifies the top users who contribute the most to positive, negative, and neutral conversations within the chat. This helps in understanding individual communication styles and their overall impact on the group's sentiment.

* **📝 Word Analysis:** Delve into the vocabulary associated with different sentiments. This feature generates visualizations of the most frequently used words for positive, neutral, and negative messages. By filtering out common "stop words" (e.g., "the", "and"), it highlights meaningful terms that drive the emotional context of the chat.

* **🔒 Privacy-Focused:** A cornerstone of this application is its commitment to your privacy. All chat data processing, analysis, and visualization are performed **client-side within your web browser**. This means your sensitive chat information is **never uploaded to any servers**, stored externally, or transmitted over the internet. Your data remains entirely on your local machine, ensuring complete confidentiality and security.

---

## 🚀 Getting Started

Follow these steps to set up and run the WhatsApp Chat Sentiment Analyzer on your local machine.

### Prerequisites

Ensure you have Python 3.7 or higher installed.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/Udhay-Brahmi/WACSA-Whats-App-Chat-Sentiment-Analyzer.git](https://github.com/Udhay-Brahmi/WACSA-Whats-App-Chat-Sentiment-Analyzer.git)
    cd WACSA-Whats-App-Chat-Sentiment-Analyzer
    ```

    (Replace `your-username` with your actual GitHub username)

2.  **Install dependencies:**
    It's highly recommended to use a virtual environment.
    
    ```bash
    # Create a virtual environment
    python -m venv venv

    # Activate the virtual environment
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate

    # Install required packages
    pip install -r requirements.txt
    ```

    The `requirements.txt` file ensures all necessary libraries (Gradio, pandas, numpy, nltk, matplotlib, seaborn, scipy, pyarrow, etc.) are installed.

### Running the Application

Once the dependencies are installed, you can launch the Gradio application:

```bash
python app.py
```
---

## 🎬 Demo Video 
[![WhatsApp Chat Sentiment Analyzer Demo](https://img.youtube.com/vi/M8AlfcW0M70/0.jpg)](https://www.youtube.com/watch?v=M8AlfcW0M70&list=PL0JlkXkl7laZ9cC5h8QM5tZE5wj9VOrYS&index=1&t=2s)

---

## 📞 Point of Contact

For any questions or collaborations, feel free to reach out:

**Udhay Brahmi**
* **Email:** udhaybrahmi786@gmail.com
* **Affiliation:** MS by Research (CSE), IIT Bombay
