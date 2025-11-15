📊 Customer Review Sentiment Analysis System
An AI-powered sentiment analysis application built during a 2-hour ML workshop, leveraging state-of-the-art transformer models to analyze customer reviews at scale.


🎯 Project Overview
This project demonstrates end-to-end machine learning engineering by building a production-ready sentiment analysis system. The application analyzes customer reviews to determine sentiment (positive/negative) with high accuracy, making it valuable for businesses to understand customer feedback efficiently.
Built for: ML Workshop & Portfolio Development
Development Time: 2 hours (MVP) → Expandable to full production system
Model: DistilBERT (Hugging Face Transformers)

🤖 AI-Assisted Development Journey
This project was developed using AI pair programming tools to accelerate learning and implementation:
Tools Used:

GitHub Copilot: foldered in sentimental_analysis_ghcp
Jules (Cursor AI):foldered in workshop-sentiment
Claude (Anthropic): Project planning, documentation, and technical mentorship

Learning Approach:
Rather than just copying code, I used AI agents as learning accelerators to:

✅ Understand transformer architecture and NLP preprocessing techniques
✅ Learn Streamlit framework and modern web app development
✅ Master proper code structure, error handling, and documentation
✅ Implement industry best practices for ML deployment
✅ Debug issues and optimize performance in real-time

Key Insight: AI tools didn't replace learning—they enabled me to build production-quality software while deeply understanding each component. Every line of code was reviewed, tested, and comprehended.

✨ Features
Core Functionality:

Single Review Analysis: Instant sentiment prediction with confidence scores
Bulk CSV Processing: Analyze thousands of reviews with progress tracking
Interactive Dashboard: Visual insights with charts and metrics
Export Results: Download analyzed data as CSV for further processing

Technical Highlights:

🔥 Pre-trained DistilBERT Model: 90%+ accuracy on real-world reviews
⚡ Fast Processing: Handle 10,000+ reviews efficiently
🎨 Professional UI: Clean, responsive design with real-time feedback
📊 Data Visualization: Interactive charts using Plotly
🛡️ Error Handling: Robust validation and graceful error management


🚀 Quick Start
Installation
bash# Clone the repository
git clone https://github.com/yourusername/sentiment-analysis.git
cd sentiment-analysis

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
The app will open automatically at http://localhost:8501
Requirements
txtstreamlit==1.28.0
pandas==2.1.0
transformers==4.35.0
torch==2.1.0
plotly==5.17.0
wordcloud==1.9.2
matplotlib==3.8.0

💡 Usage
Single Review Analysis

Navigate to the "📝 Single Review" tab
Paste or type a customer review
Click "🔍 Analyze Sentiment"
View sentiment, confidence score, and processed text

Bulk CSV Analysis

Go to "📊 Bulk Analysis" tab
Upload a CSV file with a review or text column
Click "🚀 Analyze All Reviews"
Explore visual insights and download results

Sample CSV Format:
csvreview
"Great product! Highly recommend. Fast shipping."
"Terrible experience. Product broke after 2 days."
"It's okay, nothing special. Average quality."

🏗️ Project Structure
sentiment-analysis/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── sample_reviews.csv     # Example dataset
└── screenshots/           # UI screenshots for documentation

🧠 Technical Deep Dive
Model Architecture

Base Model: DistilBERT (distilled version of BERT)
Fine-tuned on: Stanford Sentiment Treebank (SST-2)
Framework: PyTorch + Hugging Face Transformers
Accuracy: 90%+ on benchmark datasets

Text Preprocessing Pipeline

Lowercase conversion
URL and special character removal
Whitespace normalization
Tokenization via DistilBERT tokenizer

Performance Metrics

Single Review: < 1 second
1,000 Reviews: ~ 20-30 seconds
10,000 Reviews: ~ 3-5 minutes


🎓 Learning Outcomes
Through this project, I gained hands-on experience with:
Machine Learning & NLP:

Transformer architecture (BERT family)
Sentiment classification techniques
Pre-trained model fine-tuning
Text preprocessing best practices

Software Engineering:

Web application development with Streamlit
API integration (Hugging Face)
Error handling and validation
Code documentation and structure

Data Science:

Data visualization with Plotly
Pandas for data manipulation
Batch processing optimization
Results export and reporting

AI-Assisted Development:

Effective prompting for AI coding tools
Code review and debugging with AI
Balancing AI assistance with deep understanding
Production-ready code generation


🚧 Future Enhancements
Phase 1 (Planned):

 Custom model training on domain-specific data
 Advanced preprocessing (contractions, slang, emojis)
 Aspect-based sentiment analysis
 Multi-language support

Phase 2 (Advanced):

 Real-time Twitter/social media integration
 Sentiment trend analysis over time
 User authentication and saved analyses
 RESTful API deployment
 Docker containerization
 Cloud deployment (AWS/Heroku)

Phase 3 (Production):

 Database integration (PostgreSQL)
 Caching layer for performance
 A/B testing framework
 Advanced analytics dashboard
 Model versioning and monitoring


📊 Results & Demo
Sample Analysis Results:

Positive Reviews: 67%
Negative Reviews: 33%
Average Confidence: 88.5%

Screenshots:
<img width="1919" height="924" alt="image" src="https://github.com/user-attachments/assets/c15b2331-4366-41aa-a58a-95e2c6bd99f7" />
<img width="1919" height="930" alt="image" src="https://github.com/user-attachments/assets/667f28ab-ed27-4de7-9d81-5417f8289832" />
<img width="1919" height="921" alt="image" src="https://github.com/user-attachments/assets/2f97e1bf-2e39-41b6-879f-8e18f8934721" /> (Built using github Copilot


🤝 Contributing
This is a learning project, but feedback and suggestions are welcome! Feel free to:

Open issues for bugs or feature requests
Submit pull requests with improvements
Share your own implementations or extensions


📝 Acknowledgments
AI Development Tools:

GitHu
Jules (Cursor AI)
Claude (Anthropic) for learning support and documentation

Frameworks & Libraries:

Hugging Face Transformers for pre-trained models
Streamlit for rapid web app development
Plotly for interactive visualizations

Inspiration:
Built during a hands-on ML workshop to demonstrate practical AI engineering skills while leveraging modern AI development tools.

📄 License
MIT License - Feel free to use this project for learning and development.

👤 Author
Jason Cornelius

GitHub: Jasoncornelius-j
LinkedIn: (https://www.linkedin.com/in/jason-cornelius-a3025533b/)


🌟 Key Takeaway

"This project showcases how AI-assisted development accelerates learning without compromising understanding. By using tools like GitHub Copilot and Jules, I built production-quality software while mastering ML engineering fundamentals—demonstrating both technical capability and modern development workflows."


⭐ If you found this project helpful, please consider giving it a star!
Built with 🤖 AI assistance + 🧠 human learning + ❤️ passion for ML
